import random
import time

# === Konfiguration ===
COOLDOWN = 3  # Sekunden zwischen Runden

# === Interner Zustand ===
_last_result_time = 0
_score = {"player": 0, "computer": 0}
_game_over = False

# === Spielstatus, extern abrufbar ===
game_state = {
    "player_move": "none",
    "computer_move": "none",
    "result": "",
    "last_played": 0,
    "score": _score.copy()
}


def reset_game():
    global _score, _game_over, _last_result_time
    _score = {"player": 0, "computer": 0}
    _game_over = False
    _last_result_time = 0
    game_state.update({
        "player_move": "none",
        "computer_move": "none",
        "result": "",
        "last_played": 0,
        "score": _score.copy()
    })


def play_round(player_move: str) -> dict:
    global _last_result_time, _score, _game_over

    now = time.time()

    if _game_over:
        game_state["result"] = "Game over. Please reset to play again."
        return game_state

    if now - _last_result_time < COOLDOWN:
        game_state["result"] = f"Cooldown... wait {COOLDOWN - (now - _last_result_time):.1f}s"
        return game_state

    if player_move not in {"rock", "paper", "scissors"}:
        return {
            "error": "Invalid move",
            "player_move": player_move,
            "computer_move": "none",
            "result": "Invalid move",
            "score": _score.copy(),
            "last_played": now
        }

    computer_move = random.choice(["rock", "paper", "scissors"])

    if player_move == computer_move:
        result = "Draw"
    elif (player_move, computer_move) in [
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper")
    ]:
        result = "You Win!"
        _score["player"] += 1
    else:
        result = "Computer Wins!"
        _score["computer"] += 1

    _last_result_time = now
    game_state.update({
        "player_move": player_move,
        "computer_move": computer_move,
        "result": result,
        "last_played": now,
        "score": _score.copy()
    })

    # Siegbedingung prüfen
    if _score["player"] == 2 or _score["computer"] == 2:
        game_state["result"] += " 🎉 Game Over!"
        _game_over = True

    return game_state
