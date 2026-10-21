from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from typing import Any, Callable

from ..evidence_verifier import parse_json_object
from ..methods.backends import AnswerModel, estimate_multimodal_input_tokens
from .source_media import SourceMedia
from .store import ObservationStore, SourceCatalog
from .types import Candidate, Fact, Observation, PreprocessConfig, digest


class WorkBudgetExceeded(RuntimeError):
    pass


class ResponseValidationError(ValueError):
    """Model output remained invalid after bounded repair; not an I/O failure."""

    def __init__(self, message: str, *, responses: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.responses = responses


class VerificationInputTooLarge(WorkBudgetExceeded):
    pass


MODEL_PROTOCOL = "come-offline-observe-ground-1"


def object_schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def fixed_rows_schema(field: str, rows: list[dict]) -> dict:
    # Positional schemas force every supplied ID exactly once. A generic JSON
    # object constraint cannot prevent a model from inventing candidate IDs.
    return object_schema({field: {"type": "array", "prefixItems": rows,
        "items": False, "minItems": len(rows), "maxItems": len(rows)}})


NONEMPTY_TEXT_SCHEMA = {"type": "string", "minLength": 1}


OBSERVE_REPAIR_SCHEMA = object_schema({
    "coordinate_system": {"const": "normalized_1000"},
    "observations": {"type": "array", "items": object_schema({
        "regions": {"type": "array", "items": {"type": "array", "items": {"type": "integer"},
            "minItems": 4, "maxItems": 4}},
        "description": NONEMPTY_TEXT_SCHEMA,
        "source_refs": {"type": "array", "items": {"type": "string"}},
        "fact": object_schema({
            "subject": NONEMPTY_TEXT_SCHEMA, "predicate": NONEMPTY_TEXT_SCHEMA,
            "value": NONEMPTY_TEXT_SCHEMA, "basis": {"enum": ["visual", "textual"]},
        }),
    })},
})


def complete_observation_prefix(raw: str) -> dict | None:
    """Read complete JSON items before a broken tail, without repairing content.

    Only the explicit observation array under an already parsed coordinate
    system is eligible. Stop at the first invalid item or separator; never
    search later text for objects or invent missing quotes/coordinates.
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    position = 1
    header = {}

    def whitespace() -> None:
        nonlocal position
        while position < len(text) and text[position].isspace():
            position += 1

    try:
        while True:
            whitespace()
            key, position = decoder.raw_decode(text, position)
            if not isinstance(key, str) or key in header:
                return None
            whitespace()
            if text[position:position + 1] != ":":
                return None
            position += 1
            whitespace()
            if key == "observations":
                if header.get("coordinate_system") != "normalized_1000" or text[position:position + 1] != "[":
                    return None
                position += 1
                values = []
                while True:
                    whitespace()
                    try:
                        value, end = decoder.raw_decode(text, position)
                    except ValueError:
                        break
                    if not isinstance(value, dict):
                        break
                    values.append(value)
                    position = end
                    whitespace()
                    if text[position:position + 1] != ",":
                        break
                    position += 1
                return {"coordinate_system": "normalized_1000", "observations": values} if values else None
            header[key], position = decoder.raw_decode(text, position)
            whitespace()
            if text[position:position + 1] != ",":
                return None
            position += 1
    except ValueError:
        return None


class ObservationExtractor:
    """Question-independent extraction and source grounding only."""

    def __init__(self, model: AnswerModel, catalog: SourceCatalog, packet: SourceMedia,
                 config: PreprocessConfig) -> None:
        self.model, self.catalog, self.packet, self.config = model, catalog, packet, config
        self.ingest_usage = {"calls": 0, "input_tokens": 0, "completion_tokens": 0}
        self.last_observation_report: dict = {}

    def call(self, task: str, instruction: str, nodes: list[Candidate], payload: dict,
             validate: Callable[[dict], Any],
             *, response_schema: dict | None = None, repair_schema: dict | None = None) -> Any:
        messages = [{"role": "system", "content": f"CoME task={task}. {instruction} Return a JSON object only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                        *self.packet.node_content(nodes)]}]
        last_error = None
        invalid_responses = []
        for attempt in range(3):
            estimate = estimate_multimodal_input_tokens(messages)
            # Reserve the configured completion allowance before submitting work.
            output_limit = getattr(getattr(self.model, "config", None), "max_output_tokens", 4096)
            counter = getattr(self.model, "count_input_tokens", None)
            if counter:
                estimate = counter(messages)
            if estimate + output_limit > getattr(getattr(self.model, "config", None), "max_model_len", 32768):
                raise VerificationInputTooLarge("verification input exceeds model capacity")
            schema = response_schema if response_schema is not None else repair_schema if attempt else None
            response_format = ({"type": "json_schema", "json_schema": {
                "name": task, "strict": True, "schema": schema}}
                if schema is not None else {"type": "json_object"})
            raw = self.model.complete(messages, response_format=response_format)
            input_tokens = getattr(self.model, "last_input_tokens", None) or estimate
            output_tokens = getattr(self.model, "last_completion_tokens", None) or (len(raw) + 2) // 3
            self.ingest_usage["calls"] += 1
            self.ingest_usage["input_tokens"] += input_tokens
            self.ingest_usage["completion_tokens"] += output_tokens
            try:
                return validate(parse_json_object(raw))
            except (ValueError, TypeError, KeyError) as exc:
                last_error = exc
                invalid_responses.append(raw)
                diagnostic = self.packet.crops.root.parent / "model_validation_errors.jsonl"
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                with diagnostic.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"task": task, "attempt": attempt + 1,
                        "node_ids": [node.node_id for node in nodes],
                        "error": str(exc), "response": raw}, ensure_ascii=False) + "\n")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"Invalid response: {exc}. Return a fresh complete object matching the schema."})
        raise ResponseValidationError(f"{task} failed response validation three times: {last_error}",
                                      responses=tuple(invalid_responses))

    def observe(self, image: Candidate, store: ObservationStore) -> list[Observation]:
        asset = self.catalog.assets[image.asset_id]
        self.last_observation_report = {"policy": "validated-partial-1", "accepted": 0,
            "rejected": 0, "status": "empty", "errors": []}
        if asset["type"] != "image":
            return []
        texts = {n.node_id: n for n in self.catalog.nodes.values()
                 if n.kind == "text" and n.memory_id == image.memory_id}
        # Only original text from this memory can support an observation.
        sources = list(texts.values())[:8]
        allowed_refs = {image.node_id, *(n.node_id for n in sources)}
        existing = [o for o in store.latest.values()
                    if o.asset_id == image.asset_id and o.memory_id == image.memory_id]
        # A previous run may have grounded and stored valid rows before a later
        # invalid row made the image resumable. Keep those rows available when
        # a deterministic retry produces no new proposals.
        valid_proposals: dict[str, Observation] = {o.id: o for o in existing}
        last_rejections: list[str] = []

        def entity_id(subject: str) -> str:
            prefix = image.asset_id + ":"
            if subject.startswith(prefix) and subject[len(prefix):].strip():
                return subject
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", subject):
                return prefix + subject
            raise ValueError(f"fact.subject {subject!r} must be a short local ID such as "
                             f"'person1', or start with the exact prefix {prefix!r}")

        def parse(row: dict) -> list[Observation]:
            nonlocal last_rejections
            values = row.get("observations")
            if not isinstance(values, list):
                raise ValueError("observations must be a list")
            if row.get("coordinate_system") != "normalized_1000":
                raise ValueError("coordinate_system must be 'normalized_1000'; both x and y use 0..1000")
            last_rejections = []
            limit = self.config.observations_per_image
            if len(values) > limit:
                last_rejections.append(f"observations has {len(values)} items; maximum is {limit}; "
                                       "return only the most useful distinct facts")
            facts = []
            local_ids = {}
            for index, value in enumerate(values):
                try:
                    fact_value = value["fact"].get("value")
                    if not isinstance(fact_value, str):
                        raise ValueError("fact.value must be semantic text, not coordinates; put boxes in regions")
                    if not fact_value.strip():
                        predicate = value["fact"].get("predicate")
                        embedded = re.fullmatch(r"(is|has)[_ ]+(.+)", predicate.strip(), re.I) if isinstance(predicate, str) else None
                        property_text = embedded.group(2).replace("_", " ").strip() if embedded else ""
                        if not property_text or property_text.split()[-1] in {"at", "of", "with", "than"}:
                            raise ValueError("fact.value must be nonempty semantic text; for predicate='is_blue', "
                                             "use predicate='is', value='blue'")
                        # The model sometimes puts a complete property in the
                        # predicate and leaves value blank. Split only that
                        # explicit text; grounding still checks the image.
                        value = {**value, "fact": {**value["fact"],
                            "predicate": embedded.group(1).lower(), "value": property_text}}
                        fact_value = property_text
                    if fact_value.lstrip().startswith("["):
                        try:
                            encoded_value = json.loads(fact_value)
                        except ValueError:
                            pass
                        else:
                            if isinstance(encoded_value, list):
                                raise ValueError("fact.value must be semantic text, not a stringified array; put boxes in regions")
                    fact = Fact(**value["fact"])
                    local_ids[fact.subject] = entity_id(fact.subject)
                    facts.append((index, value, fact))
                except (ValueError, TypeError, KeyError) as exc:
                    last_rejections.append(f"observation[{index}]: {exc}")
            result = []
            for index, value, fact in facts:
                try:
                    observation = parse_observation(value, fact, local_ids)
                    if observation is None:
                        continue
                    if any(o.id == observation.id for o in result):
                        raise ValueError("duplicate observation ID")
                    if len(result) >= limit:
                        continue
                    result.append(observation)
                    if observation.id in valid_proposals or len(valid_proposals) < self.config.observations_per_image:
                        valid_proposals[observation.id] = observation
                except (ValueError, TypeError, KeyError) as exc:
                    last_rejections.append(f"observation[{index}]: {exc}")
            if last_rejections:
                raise ValueError("; ".join(last_rejections) +
                    ". Correct or OMIT the invalid observations; preserve independently valid facts. "
                    "Text read from pixels uses basis='visual'.")
            return result

        def parse_observation(value: dict, fact: Fact, local_ids: dict) -> Observation | None:
            description = value.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("description must be nonempty semantic text")
            fact = replace(fact, subject=local_ids[fact.subject],
                           value=local_ids.get(fact.value, fact.value))
            refs = tuple(sorted({image.node_id, *value.get("source_refs", [])}))
            if not set(refs) <= allowed_refs:
                raise ValueError("source_refs must name supplied original evidence")
            regions = []
            for box in value["regions"]:
                if (len(box) != 4 or any(type(v) is not int for v in box) or
                    not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
                    raise ValueError(f"invalid normalized bbox {box!r}; require "
                                     "0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000")
                x1, y1, x2, y2 = box
                regions.append((x1 * asset["width"] // 1000, y1 * asset["height"] // 1000,
                    (x2 * asset["width"] + 999) // 1000, (y2 * asset["height"] + 999) // 1000))
            regions = tuple(regions)
            oid = "obs:" + digest([image.memory_id, image.asset_id, asdict(fact), regions])[:24]
            previous = store.latest.get(oid)
            observation = Observation(oid, image.memory_id, image.asset_id, regions,
                description, refs, previous.version + 1 if previous else 1,
                fact, asset["sha256"])
            self.catalog.validate(observation)
            if previous and replace(observation, version=previous.version) == previous:
                return None
            return observation
        repair_schema = {**OBSERVE_REPAIR_SCHEMA, "properties": {
            **OBSERVE_REPAIR_SCHEMA["properties"], "observations": {
                **OBSERVE_REPAIR_SCHEMA["properties"]["observations"],
                "maxItems": self.config.observations_per_image}}}
        try:
            proposed = self.call("observe",
            "Extract a few independently locatable reusable visual facts covering main entities, interactions "
            "and readable text. Each description must express one fact, not an answer to a question. "
            "Use normalized integer bboxes: BOTH axes range from 0 to 1000, regardless of image dimensions. "
            "The full image box is [0,0,1000,1000]. The host converts these to original pixel coordinates. "
            "Relations may use multiple boxes. For fact.subject, use a short local "
            "entity ID such as 'person1', 'cup1', or 'text1'; the host adds the image namespace. "
            "Reuse supplied existing entity IDs when referring to the same entity. "
            "Do not use a node_id or an asset_id as fact.subject. "
            "Never infer identity or ownership from appearance. "
            "Such facts require basis='textual' and source_refs pointing to original supporting text. "
            "OCR, printed labels, logos, and readable signs are observed pixels: use basis='visual'. "
            "Do not label a fact 'textual' merely because the image contains writing. "
            "If no original text supports an identity/ownership claim, omit that claim. "
            "The description and fact.value must express the observed fact in words. "
            "Every fact needs a nonempty value. For example, write predicate='is', value='blue' "
            "rather than predicate='is_blue', value=''. "
            f"Return at most {self.config.observations_per_image} distinct observations, ordered by usefulness. "
            "Keep descriptions concise. Do not pad values with repeated characters; omit unreadable text. "
            "Never put coordinates in fact.value; put boxes only in regions. "
            "Output {coordinate_system:'normalized_1000',observations:[{regions:[[x1,y1,x2,y2]],description,source_refs:[node_id],"
            "fact:{subject,predicate,value,basis:'visual'|'textual'}}]}. Empty observations are allowed.",
            [image, *sources], {"asset_id": image.asset_id, "width": asset["width"],
                "entity_id_prefix": image.asset_id + ":",
                "height": asset["height"], "max_observations": self.config.observations_per_image,
                "existing": [asdict(o) for o in existing]}, parse,
            repair_schema=repair_schema)
        except (ResponseValidationError, VerificationInputTooLarge) as exc:
            # A malformed tail must not erase earlier complete facts. Recover
            # only JSON-complete items, then apply the SAME semantic, reference,
            # region and grounding checks used for an ordinary response.
            for raw in getattr(exc, "responses", ()):
                try:
                    json.loads(raw)
                except ValueError:
                    recovered = complete_observation_prefix(raw)
                    if recovered is not None:
                        try:
                            parse(recovered)
                        except (ValueError, TypeError, KeyError):
                            pass  # parse retained only independently valid proposals
            # Preserve only individually validated proposals and still ground
            # them below. Invalid rows never enter the observation store.
            proposed = list(valid_proposals.values())
            self.last_observation_report.update(status="partial" if proposed else "rejected_all",
                rejected=len(last_rejections), errors=[str(exc)])
        if not proposed:
            return []
        proposed_nodes = [self.catalog.candidate(o) for o in proposed]

        def grounded(row: dict) -> list[Observation]:
            rows = row.get("checks")
            if not isinstance(rows, list):
                raise ValueError("checks must be a list")
            by_id = {str(r["id"]): r for r in rows}
            if len(by_id) != len(rows) or set(by_id) != {o.id for o in proposed}:
                raise ValueError("grounding checks must cover every proposed observation once")
            accepted = []
            for observation in proposed:
                check = by_id[observation.id]
                for field in ("visible_supported", "requires_text", "text_supported"):
                    if type(check.get(field)) is not bool:
                        raise ValueError(f"{field} must be boolean")
                if check["text_supported"] and not sources:
                    raise ValueError("text_supported=true requires supplied original text. "
                        "No original text was supplied. OCR read from image pixels uses "
                        "visible_supported=true, requires_text=false, text_supported=false.")
                if not check["visible_supported"]:
                    continue
                requires_text = check["requires_text"] or observation.fact.basis == "textual"
                if requires_text:
                    if not check["text_supported"] or not any(ref in texts for ref in observation.source_refs):
                        continue
                    observation = replace(observation, fact=replace(observation.fact, basis="textual"))
                self.catalog.validate(observation)
                accepted.append(observation)
            return accepted

        try:
            accepted = self.call("ground_observations",
            "Independently check each proposed observation against its crop AND original source. "
            "Do not accept a description merely because it was proposed. Check region localization and "
            "the entire asserted content. Identity/ownership requires specific original text that binds "
            "this entity. Generated captions do not count as identity text. "
            "Reading printed text from pixels is visual evidence and does not by itself require external text. "
            "requires_text means ONLY an identity/ownership claim requiring a binding in supplied original text. "
            "It does NOT mean that the image contains writing. text_supported refers ONLY to original_text_refs, "
            "never to words inside an image. For a visibly printed label, name, amount, or date, use "
            "visible_supported=true, requires_text=false, text_supported=false. "
            "Output {checks:[{id,visible_supported:bool,requires_text:bool,text_supported:bool}]}. "
            "visible_supported means the visible portion is established by the actual pixels.",
            [image, *sources, *proposed_nodes], {"proposed_ids": [o.id for o in proposed],
                "original_text_refs": [source.node_id for source in sources]}, grounded,
            response_schema=fixed_rows_schema("checks", [object_schema({
                "id": {"const": o.id}, "visible_supported": {"type": "boolean"},
                "requires_text": {"type": "boolean"},
                "text_supported": {"type": "boolean"} if sources else {"const": False},
            }) for o in proposed]))
        except (ResponseValidationError, VerificationInputTooLarge) as exc:
            self.last_observation_report["status"] = "grounding_failed"
            self.last_observation_report["errors"].append(str(exc))
            accepted = []
        self.last_observation_report["accepted"] = len(accepted)
        self.last_observation_report["rejected"] += len(proposed) - len(accepted)
        if self.last_observation_report["status"] == "partial" and accepted:
            self.last_observation_report["status"] = "validated_partial"
        if self.last_observation_report["status"] == "empty":
            self.last_observation_report["status"] = "validated" if accepted else "rejected_all"
        return accepted
