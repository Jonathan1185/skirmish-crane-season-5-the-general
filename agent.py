"""Choose a reachable tile, then a target, with the trained PyTorch policy.

We encode the acting unit, visible units, and zones once. Movement candidates read that
context through attention. Each target candidate points directly at its enemy's token.
Both decisions use legal options, with the first option breaking equal-score ties.
"""

from pathlib import Path
from heapq import heappop, heappush

import torch
from sandbox.crane import action, me, paths, roster, tile, units, visible, zone
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation
from encoder import (
    describe_move,
    describe_target,
    encode_entities,
    move_options,
    target_entity_indices,
    target_options,
)
from imitation_model import load_policy


def option_tensor(rows):
    """Add a batch dimension: one decision, several rows, several features per row."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


@torch.inference_mode()
def choose_action(model, observation):
    # Read the battlefield once. Both scorers use this same context.
    entities, enemy_indices = encode_entities(observation)
    # A mask marks real rows. During play every row is real; training also has padding.
    entity_mask = torch.ones((1, len(entities)), dtype=torch.bool)
    context = model.encode(option_tensor(entities), entity_mask)

    moves = move_options(observation)
    targets = target_options(observation)
    if not moves or not targets:
        raise ValueError("The observation must contain legal movement and target options")
    # Score destinations, keeping one legal path to each. argmax keeps the first tie.
    move_rows = option_tensor([describe_move(observation, end) for _, end in moves])
    move_mask = torch.ones((1, len(moves)), dtype=torch.bool)
    scores = model.score_moves(context, entity_mask, move_rows, move_mask)
    path, end = moves[int(scores[0].argmax())]

    # Describe targets from the tile we actually chose. The pointers connect target IDs
    # to the enemy tokens in context; -1 selects the model's learned no-target value.
    target_rows = option_tensor([describe_target(observation, end, target) for target in targets])
    pointers = torch.tensor([target_entity_indices(observation, targets, enemy_indices)])
    target_mask = torch.ones((1, len(targets)), dtype=torch.bool)
    scores = model.score_targets(context, target_rows, pointers, target_mask)
    return {"path": path, "target": targets[int(scores[0].argmax())]}


class Agent:
    """A terrain-aware team strategist whose attacks use the trained policy.

    Movement is deliberately role based: cavalry capture, footmen screen archers,
    and archers maintain range.  Once that movement is fixed, the neural policy
    chooses the best legal target from the resulting tile.
    """

    THREAT_DISTANCE = 3

    def __init__(self):
        # A model this small runs fastest on one CPU thread. The setting applies to the whole
        # process, so we set it before the first inference.
        torch.set_num_threads(1)
        self.model = load_policy(Path(__file__).with_name("imitation-policy.pt"))

    def reset(self, seed, observation: SkirmishObservation) -> None:
        """Reset this unit's local coordination state for a new match."""
        self._archer_health: dict[str, int] = {}
        allies = roster.allies(observation)
        self._ally_roles = {ally["player"]: ally["type"] for ally in allies}
        self._player_by_unit = {ally["unit_id"]: ally["player"] for ally in allies}
        self._role_players = {
            role: {player for player, player_role in self._ally_roles.items() if player_role == role}
            for role in ("cavalry", "footman")
        }
        self._last_heartbeat = {
            role: {player: 0 for player in players}
            for role, players in self._role_players.items()
        }
        self._own_type = me.unit_type(observation)
        self._last_observed_round = observation["observation"]["round"]

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        """Plan role-specific movement, then select a learned target for its endpoint."""
        self._last_observed_round = observation["observation"]["round"]
        self._record_visible_allies(observation)
        unit_type = me.unit_type(observation)
        if unit_type == "archer":
            role_order = self._archer_order(observation)
        elif unit_type == "cavalry":
            role_order = self._cavalry_order(observation)
        else:
            role_order = self._footman_order(observation)

        # The old role code supplies a sensible preferred target, but the policy
        # sees the exact selected endpoint and can make the finer attack choice.
        try:
            return {"path": role_order["path"], "target": self._target_for_path(observation, role_order["path"])}
        except (RuntimeError, ValueError):
            # Keep a valid strategic order if model inference is ever unavailable.
            return role_order

    @torch.inference_mode()
    def _target_for_path(self, observation: SkirmishObservation, path_id: int) -> int:
        """Score legal targets after following ``path_id`` chosen by the role plan."""
        entities, enemy_indices = encode_entities(observation)
        entity_mask = torch.ones((1, len(entities)), dtype=torch.bool)
        context = self.model.encode(option_tensor(entities), entity_mask)
        targets = target_options(observation)
        if not targets:
            raise ValueError("The observation must contain legal target options")
        end = tile.at_path_end(me.position(observation), path_id)
        rows = option_tensor([describe_target(observation, end, target) for target in targets])
        pointers = torch.tensor([target_entity_indices(observation, targets, enemy_indices)])
        target_mask = torch.ones((1, len(targets)), dtype=torch.bool)
        scores = self.model.score_targets(context, rows, pointers, target_mask)
        return targets[int(scores[0].argmax())]

    def _archer_order(self, observation: SkirmishObservation) -> SkirmishAction:
        # Capture and hold the closest scoring zone whenever zones are active.
        # The combat logic below remains the fallback for zone-free presets.
        if zone.zones(observation):
            return self._capture_order(observation)

        enemies = visible.enemies(observation)
        footmen = [ally for ally in visible.allies(observation) if ally["type"] == "footman"]
        if not enemies:
            return self._move_toward(observation, tile.at_center(observation))

        here = me.position(observation)
        target = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))
        attack_range = units.STATS["archer"].attack_range
        attack_paths = [
            path
            for path in action.legal_paths(observation)
            if tile.distance(tile.at_path_end(here, path), target["position"]) <= attack_range
        ]
        if not attack_paths:
            return self._move_toward(observation, target["position"], target["unit_id"])

        path_id = max(
            attack_paths,
            key=lambda path: (
                tile.distance(tile.at_path_end(here, path), target["position"]),
                min(tile.distance(tile.at_path_end(here, path), enemy["position"]) for enemy in enemies),
                -min((tile.distance(tile.at_path_end(here, path), footman["position"]) for footman in footmen), default=0),
                len(paths.decode(path)),
                -path,
            ),
        )
        return action.move(path_id, target["unit_id"], observation)

    def _cavalry_order(self, observation: SkirmishObservation) -> SkirmishAction:
        enemies = visible.enemies(observation)
        here = me.position(observation)
        target_id = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))["unit_id"] if enemies else None
        if zone.at(observation, here) is not None:
            return action.stay(target_id, observation) if target_id else action.stay()
        return self._move_toward(observation, self._capture_goal(observation), target_id)

    def _footman_order(self, observation: SkirmishObservation) -> SkirmishAction:
        # Capture and hold the closest scoring zone whenever zones are active.
        if zone.zones(observation):
            return self._capture_order(observation)

        archers = [ally for ally in visible.allies(observation) if ally["type"] == "archer"]
        enemies = visible.enemies(observation)
        if not archers and enemies:
            here = me.position(observation)
            target = min(enemies, key=lambda enemy: (enemy["hit_points"], tile.distance(here, enemy["position"])))
            return self._move_toward(observation, target["position"], target["unit_id"])

        hurt_archers = [
            archer for archer in archers
            if archer["unit_id"] in self._archer_health
            and archer["hit_points"] < self._archer_health[archer["unit_id"]]
        ]
        self._archer_health.update({archer["unit_id"]: archer["hit_points"] for archer in archers})
        threatened_archers = [
            archer for archer in archers
            if archer in hurt_archers or any(
                tile.distance(enemy["position"], archer["position"]) <= self.THREAT_DISTANCE for enemy in enemies
            )
        ]
        threats = [
            enemy for enemy in enemies if any(
                archer in hurt_archers or tile.distance(enemy["position"], archer["position"]) <= self.THREAT_DISTANCE
                for archer in threatened_archers
            )
        ]
        if threats:
            here = me.position(observation)
            target = min(
                threats,
                key=lambda enemy: (
                    min(tile.distance(enemy["position"], archer["position"]) for archer in threatened_archers),
                    tile.distance(here, enemy["position"]),
                ),
            )
            return self._move_toward(observation, target["position"], target["unit_id"])
        if archers:
            here = me.position(observation)
            archer = min(archers, key=lambda ally: tile.distance(here, ally["position"]))
            return self._move_toward(observation, archer["position"])
        return self._move_toward(observation, tile.at_center(observation))

    def _capture_order(self, observation: SkirmishObservation) -> SkirmishAction:
        here = me.position(observation)
        enemies = visible.enemies(observation)
        target_id = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))["unit_id"] if enemies else None
        if zone.at(observation, here) is not None:
            return action.stay(target_id, observation) if target_id else action.stay()
        return self._move_toward(observation, self._capture_goal(observation), target_id)

    @staticmethod
    def _capture_goal(observation: SkirmishObservation) -> AxialPosition:
        """Return the capture-zone center nearest to this unit, if one exists."""
        areas = zone.zones(observation)
        here = me.position(observation)
        return (
            min(areas, key=lambda area: tile.distance(here, area["center"]))["center"]
            if areas
            else tile.at_center(observation)
        )

    def _record_visible_allies(self, observation: SkirmishObservation) -> None:
        current_round = observation["observation"]["round"]
        for ally in visible.allies(observation):
            player = self._player_by_unit.get(ally["unit_id"])
            if player is not None:
                role = self._ally_roles[player]
                if role in self._last_heartbeat:
                    self._last_heartbeat[role][player] = current_round

    def _all_role_lost(self, role: str, observation: SkirmishObservation) -> bool:
        players = self._role_players[role]
        if not players:
            return False
        current_round = observation["observation"]["round"]
        return all(self._last_heartbeat[role][player] < current_round - 2 for player in players)

    def _move_toward(self, observation: SkirmishObservation, goal: AxialPosition, target_id: str | None = None) -> SkirmishAction:
        here = me.position(observation)
        path_id = self._pathfind_path(observation, goal)
        if path_id is None:
            path_id = min(
                action.legal_paths(observation),
                key=lambda path: (tile.distance(tile.at_path_end(here, path), goal), -len(paths.decode(path)), path),
            )
        return action.move(path_id, target_id, observation) if target_id else action.move(path_id)

    def _pathfind_path(self, observation: SkirmishObservation, goal: AxialPosition) -> int | None:
        route = self._a_star_route(observation, me.position(observation), goal)
        if route is None:
            return None
        legal_paths = set(action.legal_paths(observation))
        for length in range(min(paths.MAX_STEPS, len(route)), 0, -1):
            path_id = paths.encode(route[:length])
            if path_id in legal_paths:
                return path_id
        return 0

    def _a_star_route(self, observation: SkirmishObservation, start: AxialPosition, goal: AxialPosition) -> tuple[int, ...] | None:
        visible_units = [*visible.allies(observation), *visible.enemies(observation)]
        occupied = {(unit["position"]["q"], unit["position"]["r"]) for unit in visible_units}
        start_key, goal_key = (start["q"], start["r"]), (goal["q"], goal["r"])
        if goal_key in occupied or not self._is_passable(observation, goal):
            destinations = {
                (neighbor["q"], neighbor["r"])
                for neighbor in tile.neighbors(goal).values()
                if self._is_passable(observation, neighbor) and (neighbor["q"], neighbor["r"]) not in occupied
            }
        else:
            destinations = {goal_key}
        if not destinations:
            return None

        def heuristic(position: tuple[int, int]) -> int:
            current: AxialPosition = {"q": position[0], "r": position[1]}
            return min(tile.distance(current, {"q": q, "r": r}) for q, r in destinations)

        frontier: list[tuple[int, int, int, tuple[int, int]]] = [(heuristic(start_key), 0, 0, start_key)]
        costs, previous, sequence = {start_key: 0}, {}, 0
        while frontier:
            _, cost_so_far, _, current = heappop(frontier)
            if cost_so_far != costs.get(current):
                continue
            if current in destinations:
                route: list[int] = []
                while current != start_key:
                    current, direction = previous[current]
                    route.append(direction)
                return tuple(reversed(route))
            current_position: AxialPosition = {"q": current[0], "r": current[1]}
            for direction, neighbor in tile.neighbors(current_position).items():
                neighbor_key = (neighbor["q"], neighbor["r"])
                if neighbor_key in occupied or not self._is_passable(observation, neighbor):
                    continue
                next_cost = cost_so_far + self._terrain_cost(observation, neighbor)
                if next_cost >= costs.get(neighbor_key, float("inf")):
                    continue
                costs[neighbor_key] = next_cost
                previous[neighbor_key] = (current, direction)
                sequence += 1
                heappush(frontier, (next_cost + heuristic(neighbor_key), next_cost, sequence, neighbor_key))
        return None

    @staticmethod
    def _is_passable(observation: SkirmishObservation, position: AxialPosition) -> bool:
        return tile.terrain_at(observation, position)["terrain"] not in {"water", "void"}

    @staticmethod
    def _terrain_cost(observation: SkirmishObservation, position: AxialPosition) -> int:
        ground = tile.terrain_at(observation, position)
        return (2 if ground["terrain"] == "hill" else 1) + {"forest": 1, "marsh": 2}.get(ground["feature"], 0)

    def chat(self, inbox: list[dict]) -> list[dict]:
        """Exchange role heartbeats so replacements know when to capture."""
        for message in inbox:
            sender, text = message.get("from"), message.get("text")
            role = self._ally_roles.get(sender)
            if role not in self._last_heartbeat or not isinstance(text, str):
                continue
            parts = text.split(":")
            if len(parts) != 3 or parts[:2] != ["crane-status", role]:
                continue
            try:
                reported_round = int(parts[2])
            except ValueError:
                continue
            self._last_heartbeat[role][sender] = max(self._last_heartbeat[role][sender], reported_round)
        return [{"to": None, "text": f"crane-status:{self._own_type}:{self._last_observed_round}"}]
