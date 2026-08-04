from typing import TypedDict
from dotenv import load_dotenv
from datetime import date
import os
import requests
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


class ScoutState(TypedDict):
    player_name: str
    player_data: dict
    position_category: str
    scouting_report: str
    retry_count: int              # how many times check_quality has sent this report back for a redo
    quality_feedback: str         # what check_quality found wrong last time — empty string means nothing wrong yet
    debug_force_fail: bool        # test-only switch: forces the first attempt to be treated as broken, to prove the loop fires


# Fetches real player data from TheSportsDB API and extracts key fields
def fetch_player(state: ScoutState) -> ScoutState:
    url = f"https://www.thesportsdb.com/api/v1/json/123/searchplayers.php?p={state['player_name']}"
    response = requests.get(url)
    players = response.json().get("player", [])
    if not players:
        return {**state, "player_data": {}}
    player = players[0]
    return {
        **state,
        "player_data": {
            "strTeam": player.get("strTeam", "Unknown"),
            "strPosition": player.get("strPosition", "Unknown"),
            "strNationality": player.get("strNationality", "Unknown"),
            "dateBorn": player.get("dateBorn", "Unknown"),
        },
    }


# Maps strPosition string to one of five categories: forward, midfielder, defender, goalkeeper, or manager
def classify_position(state: ScoutState) -> ScoutState:
    position = state["player_data"].get("strPosition", "").lower()
    wing_back_keywords = ["wing-back", "wingback", "wing back"]   # a wing-back is a defender, not a forward — must be checked before the generic "wing" match below
    goalkeeper_keywords = ["goalkeeper", "keeper", "gk"]
    manager_keywords = ["manager", "coach", "head coach"]
    forward_keywords = ["forward", "striker", "winger", "wing", "attacking", "centre-forward", "st ", "cf ", " lw", " rw"]
    midfielder_keywords = ["midfielder", "midfield", "central mid", "box-to-box", " cam", " cdm", " cm"]
    defender_keywords = ["defender", "back", "centre-back", "fullback", "full-back", "sweeper", "libero"]

    for kw in wing_back_keywords:            # checked first, so "Left Wing-Back" doesn't fall into the forward branch below
        if kw in position:
            return {**state, "position_category": "defender"}
    for kw in goalkeeper_keywords:
        if kw in position:
            return {**state, "position_category": "goalkeeper"}
    for kw in manager_keywords:
        if kw in position:
            return {**state, "position_category": "manager"}
    for kw in forward_keywords:
        if kw in position:
            return {**state, "position_category": "forward"}
    for kw in midfielder_keywords:
        if kw in position:
            return {**state, "position_category": "midfielder"}
    for kw in defender_keywords:
        if kw in position:
            return {**state, "position_category": "defender"}

    return {**state, "position_category": "midfielder"}


def route_by_position(state: ScoutState) -> str:
    return state["position_category"]


BASE_REQUIRED_SECTIONS = ["SPECIALITY", "OVERALL CONCLUSION"]          # every report, any position, needs these
POSITION_REQUIRED_SECTIONS = {                                         # extra sections required for specific positions
    "midfielder": ["DEFENSIVE CONTRIBUTION, DUELS AND DISCIPLINE"],    # the exact gap Bruno Guimarães' report had
}
MAX_RETRIES = 2                                                        # cap so a stubborn LLM can't loop forever


# Reads the report a scout node just wrote and decides: good enough, or send it back
def check_quality(state: ScoutState) -> ScoutState:
    report = state.get("scouting_report", "")
    retry_count = state.get("retry_count", 0)
    required = BASE_REQUIRED_SECTIONS + POSITION_REQUIRED_SECTIONS.get(state.get("position_category", ""), [])

    if state.get("debug_force_fail") and retry_count == 0:      # test-only: pretend attempt 1 is broken, no matter what
        missing = ["OVERALL CONCLUSION"]
    else:
        missing = [section for section in required if section not in report]   # sections it should have but doesn't

    if not missing or retry_count >= MAX_RETRIES:           # nothing wrong, or we've already given it enough chances
        return {**state, "quality_feedback": ""}

    return {
        **state,
        "retry_count": retry_count + 1,                                           # count this attempt against the budget
        "quality_feedback": f"missing required section(s): {', '.join(missing)}",  # this message travels backward
    }


# THE REVERSE EDGE'S BRAIN: routes back to whichever scout node just ran, or forward to END
def route_after_quality_check(state: ScoutState) -> str:
    if state.get("quality_feedback"):        # check_quality found a problem
        return state["position_category"]    # go back to the SAME scout node that wrote the bad report
    return "end"                             # nothing wrong, let it finish


def _llm():
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)


def _today():
    return date.today().strftime("%Y-%m-%d")


# Stops the LLM from defaulting to generic scouting clichés — forces it to ground the SPECIALITY line
# in what this player's actual listed position implies about their real-world reputation
def _speciality_grounding(raw_position: str) -> str:
    return (
        f'Do NOT default to generic scouting clichés (e.g. "dynamic and creative", "exceptional vision") unless '
        f'they are genuinely the single most defining thing about THIS specific player. Their actual listed '
        f'position is "{raw_position}" — let that concretely inform what they are most likely known for (a '
        f'player in a defensive-leaning role is far more likely known for ball-winning, tempo control, and '
        f'aggression than for being a flashy creator, unless there is strong reason otherwise). Be willing to '
        f"say a player's defining trait is something unglamorous like tackling, discipline, or work rate if "
        f"that is genuinely their real identity, instead of forcing every player into the same 'creative, "
        f"vision-driven' mold."
    )


# Generates a forward-specific scouting report focused on goals, xG, pressing, and finishing
def scout_forward(state: ScoutState) -> ScoutState:
    template = PromptTemplate(
        input_variables=["player_name", "team", "nationality", "date_born", "today", "speciality_grounding"],
        template="""You are an elite football scout. Today's date is {today}. Calculate the player's current age from their date of birth.

Player: {player_name}
Club: {team}
Nationality: {nationality}
Date of Birth: {date_born}

You MUST follow this exact structure with these exact section headers. Do not rename, reorder, or skip any section.

SPECIALITY
{speciality_grounding}
Write one punchy sentence describing this player's unique identity and defining trait as a forward (e.g. "Clinical penalty-box predator — elite positioning, lethal two-footed finishing, elite aerial threat").

GOAL-SCORING ABILITY AND FINISHING QUALITY (X/10)
Assessment of their finishing, consistency, and variety of goals.

EXPECTED GOALS (xG) PERFORMANCE AND SHOT SELECTION (X/10)
Assessment of their xG output, shot quality, and decision-making in front of goal.

PRESSING INTENSITY AND OFF-THE-BALL MOVEMENT (X/10)
Assessment of their work rate, pressing triggers, and movement without the ball.

FINISHING TECHNIQUE IN AND AROUND THE BOX (X/10)
Assessment of their technique, composure, and clinical ability inside the box.

OVERALL CONCLUSION
A paragraph summarising the player's overall style, effectiveness, key strengths, areas for improvement, and your verdict on their level and value.

For each attribute write 2 to 3 sentences of detailed analysis. Be specific and analytical, not generic.""",
    )
    d = state["player_data"]
    prompt = template.format(
        player_name=state["player_name"],
        team=d.get("strTeam", "Unknown"),
        nationality=d.get("strNationality", "Unknown"),
        date_born=d.get("dateBorn", "Unknown"),
        today=_today(),
        speciality_grounding=_speciality_grounding(d.get("strPosition", "Unknown")),
    )
    if state.get("quality_feedback"):        # this line is the loop actually landing back here on a retry
        prompt += f"\n\nIMPORTANT: your previous attempt was rejected — {state['quality_feedback']}. Fix that this time."
    return {**state, "scouting_report": _llm().invoke(prompt).content}


# Generates a midfielder-specific scouting report — always covers passing AND defensive duels, the latter
# adapting its emphasis depending on whether the player's actual listed position leans defensive or attacking
def scout_midfielder(state: ScoutState) -> ScoutState:
    raw_position = state["player_data"].get("strPosition", "").lower()
    is_defensive_leaning = any(kw in raw_position for kw in ["defensive", "holding", "cdm", "anchor"])

    if is_defensive_leaning:                                     # e.g. Bruno Guimarães — "Defensive Midfield"
        defensive_guidance = "This player's listed position is defensive midfield, so weight this section heavily: assess their tackling aggression, discipline (fouls and cards picked up), and how often they win the ball back — not just whether they avoid mistakes."
    else:
        defensive_guidance = "Assess their defensive work rate, tracking back, and contribution to pressing and ball recovery, even though this isn't their primary strength."

    template = PromptTemplate(
        input_variables=["player_name", "team", "nationality", "date_born", "today", "defensive_guidance", "speciality_grounding"],
        template="""You are an elite football scout. Today's date is {today}. Calculate the player's current age from their date of birth.

Player: {player_name}
Club: {team}
Nationality: {nationality}
Date of Birth: {date_born}

You MUST follow this exact structure with these exact section headers. Do not rename, reorder, or skip any section.

SPECIALITY
{speciality_grounding}
Write one punchy sentence describing this player's unique identity and defining trait as a midfielder (e.g. "Relentless box-to-box engine — elite stamina, non-stop tackling, covers every blade of grass").

PASSING RANGE AND DISTRIBUTION QUALITY (X/10)
Assessment of their short, medium, and long passing accuracy and vision.

PRESS RESISTANCE AND COMPOSURE UNDER PRESSURE (X/10)
Assessment of how well they retain the ball and make decisions when pressed.

CREATIVITY AND ABILITY TO BREAK DEFENSIVE LINES (X/10)
Assessment of their vision, through balls, and ability to unlock defences.

DEFENSIVE CONTRIBUTION, DUELS AND DISCIPLINE (X/10)
{defensive_guidance}

OVERALL CONCLUSION
A paragraph summarising the player's overall style, effectiveness, key strengths, areas for improvement, and your verdict on their level and value.

For each attribute write 2 to 3 sentences of detailed analysis. Be specific and analytical, not generic.""",
    )
    d = state["player_data"]
    prompt = template.format(
        player_name=state["player_name"],
        team=d.get("strTeam", "Unknown"),
        nationality=d.get("strNationality", "Unknown"),
        date_born=d.get("dateBorn", "Unknown"),
        today=_today(),
        defensive_guidance=defensive_guidance,
        speciality_grounding=_speciality_grounding(d.get("strPosition", "Unknown")),
    )
    if state.get("quality_feedback"):        # this line is the loop actually landing back here on a retry
        prompt += f"\n\nIMPORTANT: your previous attempt was rejected — {state['quality_feedback']}. Fix that this time."
    return {**state, "scouting_report": _llm().invoke(prompt).content}


# Generates a goalkeeper-specific scouting report with ratings out of 10 for each attribute
def scout_goalkeeper(state: ScoutState) -> ScoutState:
    template = PromptTemplate(
        input_variables=["player_name", "team", "nationality", "date_born", "today", "speciality_grounding"],
        template="""You are an elite football scout specialising in goalkeepers. Today's date is {today}. Calculate the player's current age from their date of birth.

Player: {player_name}
Club: {team}
Nationality: {nationality}
Date of Birth: {date_born}

You MUST follow this exact structure with these exact section headers. Do not rename, reorder, or skip any section.

SPECIALITY
{speciality_grounding}
Write one punchy sentence describing this goalkeeper's unique identity and defining trait (e.g. "Elite shot-stopper with sweeper-keeper instincts — dominates his box and starts attacks with precision distribution").

SHOT STOPPING (X/10)
Assessment of their reflexes, positioning, and save technique.

COMMAND OF AREA (X/10)
Assessment of their ability to claim crosses, organise the defence, and dominate aerially.

DISTRIBUTION FROM THE BACK (X/10)
Assessment of their short passing, long kicks, and accuracy under pressure.

SWEEPER KEEPER ABILITY (X/10)
Assessment of their reading of play, rushing off the line, and one-on-one situations.

PENALTY SAVING (X/10)
Assessment of their dive technique, psychological composure, and penalty-saving record.

OVERALL CONCLUSION
A paragraph summarising the goalkeeper's overall style, effectiveness, key strengths, areas for improvement, and your verdict on their level and value.

For each attribute write 2 to 3 sentences of detailed analysis. Be specific and analytical, not generic.""",
    )
    d = state["player_data"]
    prompt = template.format(
        player_name=state["player_name"],
        team=d.get("strTeam", "Unknown"),
        nationality=d.get("strNationality", "Unknown"),
        date_born=d.get("dateBorn", "Unknown"),
        today=_today(),
        speciality_grounding=_speciality_grounding(d.get("strPosition", "Unknown")),
    )
    if state.get("quality_feedback"):        # this line is the loop actually landing back here on a retry
        prompt += f"\n\nIMPORTANT: your previous attempt was rejected — {state['quality_feedback']}. Fix that this time."
    return {**state, "scouting_report": _llm().invoke(prompt).content}


# Generates a manager-specific scouting report with ratings out of 10 for each attribute
def scout_manager(state: ScoutState) -> ScoutState:
    template = PromptTemplate(
        input_variables=["player_name", "team", "nationality", "date_born", "today", "speciality_grounding"],
        template="""You are a senior football analyst evaluating a manager. Today's date is {today}. Calculate the manager's current age from their date of birth.

Name: {player_name}
Current Club: {team}
Nationality: {nationality}
Date of Birth: {date_born}

You MUST follow this exact structure with these exact section headers. Do not rename, reorder, or skip any section.

SPECIALITY
{speciality_grounding}
Write one punchy sentence describing this manager's unique identity and defining trait (e.g. "Obsessive high-press architect — builds dominant possession systems with relentless collective pressing and positional superiority").

PREFERRED FORMATION (X/10)
Assessment of their typical tactical setup, positional structure, and how effectively they implement it.

PRESS INTENSITY (X/10)
Assessment of how aggressively the team presses, at what phase of play, and how well it is executed.

ATTACKING OR DEFENSIVE STYLE (X/10)
Assessment of their overall philosophy in and out of possession.

SQUAD ROTATION APPROACH (X/10)
Assessment of how the manager handles squad depth, fatigue, and fixture congestion.

BIG GAME RECORD (X/10)
Assessment of their performance in high-stakes matches, finals, derbies, and knockout competitions.

OVERALL CONCLUSION
A paragraph summarising the manager's philosophy, overall effectiveness, notable strengths, areas for development, and your verdict on their calibre.

For each attribute write 2 to 3 sentences of detailed analysis. Be specific and analytical, not generic.""",
    )
    d = state["player_data"]
    prompt = template.format(
        player_name=state["player_name"],
        team=d.get("strTeam", "Unknown"),
        nationality=d.get("strNationality", "Unknown"),
        date_born=d.get("dateBorn", "Unknown"),
        today=_today(),
        speciality_grounding=_speciality_grounding(d.get("strPosition", "Unknown")),
    )
    if state.get("quality_feedback"):        # this line is the loop actually landing back here on a retry
        prompt += f"\n\nIMPORTANT: your previous attempt was rejected — {state['quality_feedback']}. Fix that this time."
    return {**state, "scouting_report": _llm().invoke(prompt).content}


# Generates a defender-specific scouting report focused on tackles, aerial duels, and build-up play
def scout_defender(state: ScoutState) -> ScoutState:
    template = PromptTemplate(
        input_variables=["player_name", "team", "nationality", "date_born", "today", "speciality_grounding"],
        template="""You are an elite football scout specialising in defenders. Today's date is {today}. Calculate the player's current age from their date of birth.

Player: {player_name}
Club: {team}
Nationality: {nationality}
Date of Birth: {date_born}

You MUST follow this exact structure with these exact section headers. Do not rename, reorder, or skip any section.

SPECIALITY
{speciality_grounding}
Write one punchy sentence describing this defender's unique identity and defining trait (e.g. "Dominant aerial centre-back — elite in the tackle, commanding in the air, composed ball-player from the back").

TACKLING ABILITY AND DEFENSIVE DUELS (X/10)
Assessment of their tackling technique, timing, and success rate in one-on-one situations.

AERIAL DUEL QUALITY AND HEADING ABILITY (X/10)
Assessment of their aerial dominance, heading accuracy, and defensive headers.

BUILD-UP PLAY AND PASSING FROM THE BACK (X/10)
Assessment of their composure on the ball, passing range, and ability to start attacks.

POSITIONAL DISCIPLINE AND READING OF THE GAME (X/10)
Assessment of their positioning, anticipation, and ability to read attacking threats early.

OVERALL CONCLUSION
A paragraph summarising the player's overall style, effectiveness, key strengths, areas for improvement, and your verdict on their level and value.

For each attribute write 2 to 3 sentences of detailed analysis. Be specific and analytical, not generic.""",
    )
    d = state["player_data"]
    prompt = template.format(
        player_name=state["player_name"],
        team=d.get("strTeam", "Unknown"),
        nationality=d.get("strNationality", "Unknown"),
        date_born=d.get("dateBorn", "Unknown"),
        today=_today(),
        speciality_grounding=_speciality_grounding(d.get("strPosition", "Unknown")),
    )
    if state.get("quality_feedback"):        # this line is the loop actually landing back here on a retry
        prompt += f"\n\nIMPORTANT: your previous attempt was rejected — {state['quality_feedback']}. Fix that this time."
    return {**state, "scouting_report": _llm().invoke(prompt).content}


builder = StateGraph(ScoutState)
builder.add_node("fetch_player", fetch_player)
builder.add_node("classify_position", classify_position)
builder.add_node("scout_forward", scout_forward)
builder.add_node("scout_midfielder", scout_midfielder)
builder.add_node("scout_defender", scout_defender)
builder.add_node("scout_goalkeeper", scout_goalkeeper)
builder.add_node("scout_manager", scout_manager)
builder.add_node("check_quality", check_quality)                     # the new node — sits downstream of every scout node

builder.add_edge(START, "fetch_player")
builder.add_edge("fetch_player", "classify_position")
builder.add_conditional_edges(
    "classify_position",
    route_by_position,
    {
        "forward": "scout_forward",
        "midfielder": "scout_midfielder",
        "defender": "scout_defender",
        "goalkeeper": "scout_goalkeeper",
        "manager": "scout_manager",
    },
)

# every scout node now feeds into check_quality instead of going straight to END
builder.add_edge("scout_forward", "check_quality")
builder.add_edge("scout_midfielder", "check_quality")
builder.add_edge("scout_defender", "check_quality")
builder.add_edge("scout_goalkeeper", "check_quality")
builder.add_edge("scout_manager", "check_quality")

# THE REVERSE EDGE — check_quality can send control BACK to any scout node, not just forward to END
builder.add_conditional_edges(
    "check_quality",
    route_after_quality_check,
    {
        "end": END,
        "forward": "scout_forward",
        "midfielder": "scout_midfielder",
        "defender": "scout_defender",
        "goalkeeper": "scout_goalkeeper",
        "manager": "scout_manager",
    },
)

graph = builder.compile()

if __name__ == "__main__":
    player_name = input("Enter a player name: ")
    result = graph.invoke(
        {
            "player_name": player_name,
            "player_data": {},
            "position_category": "",
            "scouting_report": "",
            "retry_count": 0,          # starts at zero, check_quality increments it if it sends the report back
            "quality_feedback": "",    # starts empty, only gets filled in when something's wrong
        }
    )
    print("\n--- SCOUTING REPORT ---\n")
    print(result["scouting_report"])
