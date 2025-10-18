
# Battle Data Structure

Each line in the .jsonl files is a JSON object with the following top-level keys:
- `battle_id`: An identifier for the battle.
- `player_won`: (In train.jsonl only) The target variable. A boolean (true or false) indicating if the player won the battle.
- `p1_team_details`: A list of 6 dictionaries, one for each Pokémon on the player's team. The first Pokémon in the list is the one they chose to lead with.
- `p2_lead_details`: A single dictionary containing the static details of the opponent's lead Pokémon.
- `battle_timeline`: A list containing turn-by-turn summaries of the battle.


## Feature Glossary
Pokémon Details Object (found in p1_team_details and p2_lead_details):
- `name`: The species name of the Pokémon. (e.g., ""Starmie")
- `level`: The Pokémon's level. (e.g., 100, note that this is typically the case in competitive battles).
- `types`: A list of Pokémon elemental types. (e.g., ["Water", "Psychic"]).
- `base_hp`: The base HP stat of the Pokémon.
- `base_atk`: The base Attack stat of the Pokémon.
- `base_def`: The base Defense stat of the Pokémon.
- `base_spa`: The base Special Attack stat of the Pokémon.
- `base_spd`: The base Special Defense stat of the Pokémon.
- `base_spe`: The base Speed stat of the Pokémon.

Turn Summary Object (each element in battle_timeline):
- `turn`: The active turn number of the battle (e.g., 1, 2, 3…).
- `p1_pokemon_state`: An object detailing the player's active Pokémon that turn. Contains its name, current health (hp_pct), primary status (e.g., 'par'), volatile effects (e.g., ['confusion']), and a dictionary of in-battle stat boosts.
- `p1_move_details`: An object detailing the player's move that turn. Contains the move's name, type, category, base_power, accuracy, and priority. This can be null if no move was made (e.g., during a switch).
- `p2_pokemon_state`: Same structure as `p1_pokemon_state`, but for the opponent's active Pokémon.
- `p2_move_details`: Same structure as `p1_move_details`, but for the opponent's move.

## Example Entry
Here is an example of a single entry from train.jsonl:
```json
{
    "battle_id": 1,
    "player_won": true, 
    "p1_team_details": 
        [
            {"name": "jynx", "level": 100, "types": ["ice", "psychic"], "base_hp": 65, "base_atk": 50, "base_def": 35, "base_spa": 95, "base_spd": 95, "base_spe": 95}, 
            {"name": "snorlax", "level": 100, "types": ["normal", "notype"], "base_hp": 160, "base_atk": 110, "base_def": 65, "base_spa": 65, "base_spd": 65, "base_spe": 30}, 
            {"name": "exeggutor", "level": 100, "types": ["grass", "psychic"], "base_hp": 95, "base_atk": 95, "base_def": 85, "base_spa": 125, "base_spd": 125, "base_spe": 55}, 
            {"name": "tauros", "level": 100, "types": ["normal", "notype"], "base_hp": 75, "base_atk": 100, "base_def": 95, "base_spa": 70, "base_spd": 70, "base_spe": 110}, 
            {"name": "chansey", "level": 100, "types": ["normal", "notype"], "base_hp": 250, "base_atk": 5, "base_def": 5, "base_spa": 105, "base_spd": 105, "base_spe": 50}, 
            {"name": "slowbro", "level": 100, "types": ["psychic", "water"], "base_hp": 95, "base_atk": 75, "base_def": 110, "base_spa": 80, "base_spd": 80, "base_spe": 30}
        ], 
    "p2_lead_details": {"name": "alakazam", "level": 100, "types": ["notype", "psychic"], "base_hp": 55, "base_atk": 50, "base_def": 45, "base_spa": 135, "base_spd": 135, "base_spe": 120}, 
    "battle_timeline": 
        [
            {
                "turn": 1, 
                "p1_pokemon_state": {"name": "jynx", "hp_pct": 1.0, "status": "par", "effects": ["noeffect"], "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}}, 
                "p1_move_details": null, 
                "p2_pokemon_state": {"name": "alakazam", "hp_pct": 1.0, "status": "nostatus", "effects": ["noeffect"], "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}}, 
                "p2_move_details": {"name": "thunderwave", "type": "ELECTRIC", "category": "STATUS", "base_power": 0, "accuracy": 1.0, "priority": 0}
            },
            ...,
            {
                "turn": 30, 
                "p1_pokemon_state": {"name": "tauros", "hp_pct": 0.45, "status": "nostatus", "effects": ["noeffect"], "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}}, 
                "p1_move_details": {"name": "bodyslam", "type": "NORMAL", "category": "PHYSICAL", "base_power": 85, "accuracy": 1.0, "priority": 0}, 
                "p2_pokemon_state": {"name": "alakazam", "hp_pct": 0.77, "status": "slp", "effects": ["noeffect"], "boosts": {"atk": 0, "def": 0, "spa": -1, "spd": -1, "spe": 0}}, 
                "p2_move_details": {"name": "seismictoss", "type": "FIGHTING", "category": "PHYSICAL", "base_power": 1, "accuracy": 1.0, "priority": 0}
            }
        ]
}

