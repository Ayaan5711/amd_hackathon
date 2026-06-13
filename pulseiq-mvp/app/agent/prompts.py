"""LLM prompt templates for the agent."""

from app.tools.registry import get_tool_descriptions


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a survey analysis chatbot.

AVAILABLE TOOLS:
{tool_descriptions}

CURRENT DATA SCHEMA:
{schema_description}

CONVERSATION HISTORY:
{history}

USER MESSAGE: "{user_message}"

TASK:
1. Determine the user's intent from these options:
   - segment_stats: Compare metrics across segments/groups
   - trend_compare: Compare metrics over time or dimensions
   - open_text: Analyze open-text feedback/comments
   - anomaly: Find outliers or unusual patterns
   - recommend: Get actionable recommendations
   - clarify: User needs to provide more information
   - general: General question not requiring tools

2. If a tool is needed, extract the required parameters.

Respond in this exact JSON format:
{{
    "intent": "one_of_the_above",
    "reasoning": "Brief explanation of why this intent was chosen",
    "tool_calls": [
        {{
            "tool_name": "tool_name",
            "arguments": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
    ],
    "clarification_needed": false,
    "clarification_question": null,
    "clarification_options": []
}}

If clarification is needed, set clarification_needed to true and provide:
- clarification_question: What to ask the user
- clarification_options: List of suggested answers (if applicable)
- tool_calls: Empty list

If referring to previous context (e.g., "which department was that?"), use the conversation history to infer the intent.
"""


SYNTHESIS_PROMPT = """You are a survey analysis assistant. Synthesize tool results into a clear, helpful response.

USER MESSAGE: "{user_message}"

TOOL RESULTS:
{tool_results}

CONVERSATION CONTEXT:
{history}

TASK:
1. Provide a clear, direct answer to the user's question
2. Include specific numbers and insights from the tool results
3. Highlight the most important findings
4. Suggest 2-3 natural follow-up questions the user might ask

Respond in this exact JSON format:
{{
    "narrative": "Your detailed response here. Be conversational but precise. Include key numbers.",
    "key_insights": [
        "Key insight 1 with specific number",
        "Key insight 2 with specific number"
    ],
    "follow_up_suggestions": [
        "Suggested follow-up question 1?",
        "Suggested follow-up question 2?",
        "Suggested follow-up question 3?"
    ],
    "evidence": {{
        "main_finding": "Primary finding with number",
        "supporting_data": "Additional context"
    }}
}}

Keep the narrative concise (3-5 sentences) but informative. Focus on actionable insights.
"""


CLARIFICATION_PROMPT = """The user asked: "{user_message}"

We need clarification because: {reason}

Generate a helpful clarification question and suggest options if applicable.

Respond in JSON format:
{{
    "question": "Your clarification question",
    "suggestions": ["option1", "option2", "option3"]
}}
"""


def build_intent_prompt(
    user_message: str,
    schema: dict,
    history: list[dict[str, str]]
) -> str:
    """Build the intent classification prompt."""
    tool_descriptions = get_tool_descriptions()
    
    # Build schema description
    schema_desc = []
    for col, info in schema.items():
        schema_desc.append(
            f"- {col} ({info.get('type', 'unknown')}): "
            f"{info.get('n_unique', 0)} unique values, "
            f"examples: {', '.join(info.get('sample_values', [])[:3])}"
        )
    
    # Build history string
    history_str = ""
    if history:
        for msg in history[-4:]:  # Last 4 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
    else:
        history_str = "No previous conversation."
    
    return INTENT_CLASSIFICATION_PROMPT.format(
        tool_descriptions=tool_descriptions,
        schema_description="\n".join(schema_desc),
        user_message=user_message,
        history=history_str
    )


def build_synthesis_prompt(
    user_message: str,
    tool_results: list[dict],
    history: list[dict[str, str]]
) -> str:
    """Build the response synthesis prompt."""
    # Format tool results
    results_str = ""
    for i, result in enumerate(tool_results, 1):
        results_str += f"\nTool {i}: {result.get('tool_name', 'unknown')}\n"
        if result.get('success'):
            results_str += f"Result: {result.get('result', {})}\n"
        else:
            results_str += f"Error: {result.get('error', 'Unknown error')}\n"
    
    # Build history string
    history_str = ""
    if history:
        for msg in history[-2:]:  # Last 2 exchanges
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
    else:
        history_str = "No previous conversation."
    
    return SYNTHESIS_PROMPT.format(
        user_message=user_message,
        tool_results=results_str,
        history=history_str
    )


# =============================================================================
# Governance pack - investigation graph prompts
# =============================================================================

ORCHESTRATOR_PROMPT = """You are the orchestrator for an AI-governance audit investigation.

A batch of {total_entries} AI-assistant interaction log entries has been triaged with cheap, \
no-LLM signals (Presidio PII scan + heuristic prefilters):

- {has_pii} entries contain PII (Presidio)
- {injection_suspect} entries show signs of a prompt-injection attempt
- {compliance_suspect} entries may violate financial/medical advice policy
- {has_context} entries have retrieved policy context (candidates for a groundedness check)

Based on this triage, {dispatch_plan_size} specialist reviews will be dispatched to the \
flagged subset (security, compliance, hallucination specialists). PII risk is scored \
directly from the Presidio triage results and needs no specialist call.

Briefly explain the investigation strategy for this run and which categories are the \
highest priority.

Respond with JSON only, no other text:
{{"rationale": "<2-3 sentence investigation strategy>", "priority_categories": ["security", "compliance", "hallucination"]}}
"""


def build_orchestrator_prompt(triage_summary: dict[str, int]) -> str:
    """Build the orchestrator's investigation-strategy prompt from triage summary stats."""
    return ORCHESTRATOR_PROMPT.format(**triage_summary)


# =============================================================================
# Governance pack - "talk to results" chat prompts
# =============================================================================

GOVERNANCE_CHAT_INTENT_PROMPT = """You are an intent classifier for an AI-governance audit assistant. \
The user is asking questions about the results of a completed investigation of \
{total_entries} AI interaction log entries, of which {total_flagged} were flagged for \
specialist review.

INVESTIGATION SUMMARY:
- Findings by category: {findings_by_category}
- Risk distribution: {risk_distribution}
- Overall risk score (0-100): {overall_risk_score}

AVAILABLE TOOLS:
{tool_descriptions}

CONVERSATION HISTORY:
{history}

USER MESSAGE: "{user_message}"

TASK:
1. Determine whether answering this message requires calling one of the tools above.
2. If so, select the tool(s) and extract their arguments (e.g. category, log_id) from the \
message or conversation history.
3. If the message is general conversation that doesn't need a tool (e.g. a greeting or a \
question about what you can do), return an empty tool_calls list.

Respond in this exact JSON format:
{{
    "intent": "tool_use" or "general",
    "reasoning": "Brief explanation of why this intent was chosen",
    "tool_calls": [
        {{"tool_name": "tool_name", "arguments": {{"param1": "value1"}}}}
    ],
    "clarification_needed": false,
    "clarification_options": []
}}

If the user refers to previous context (e.g. "why was that one flagged?"), use the \
conversation history to infer the log_id or category.
"""


GOVERNANCE_CHAT_SYNTHESIS_PROMPT = """You are an AI-governance audit assistant. Synthesize the \
tool results below into a clear, helpful response about this investigation's findings.

USER MESSAGE: "{user_message}"

TOOL RESULTS:
{tool_results}

CONVERSATION CONTEXT:
{history}

TASK:
1. Directly answer the user's question, citing specific log_ids, scores, and counts from the \
tool results.
2. Highlight the most important finding.
3. Suggest 2-3 natural follow-up questions about the investigation.

Respond in this exact JSON format:
{{
    "narrative": "Your response here. Be conversational but precise, and include key numbers and log_ids.",
    "follow_up_suggestions": [
        "Suggested follow-up question 1?",
        "Suggested follow-up question 2?"
    ],
    "evidence": {{
        "main_finding": "Primary finding with number",
        "supporting_data": "Additional context"
    }}
}}

Keep the narrative concise (2-4 sentences) but informative.
"""


def _format_tool_descriptions(tool_registry: list[dict]) -> str:
    """Format an MCP-shaped tool registry (list of {name, description, parameters}) for a prompt."""
    lines = []
    for tool in tool_registry:
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])
        param_str = ", ".join(
            f"{p}: {info.get('type', 'any')}{' (required)' if p in required else ''}"
            for p, info in params.items()
        )
        lines.append(f"- {tool['name']}({param_str}): {tool['description']}")
    return "\n".join(lines)


def _format_history(history: list[dict[str, str]], limit: int) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-limit:]:
        lines.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
    return "\n".join(lines)


def _format_tool_results(tool_results: list[dict]) -> str:
    lines = []
    for i, result in enumerate(tool_results, 1):
        lines.append(f"\nTool {i}: {result.get('tool_name', 'unknown')}")
        if result.get("success"):
            lines.append(f"Result: {result.get('result', {})}")
        else:
            lines.append(f"Error: {result.get('error', 'Unknown error')}")
    return "\n".join(lines)


def build_governance_chat_intent_prompt(
    user_message: str,
    tool_registry: list[dict],
    investigation_summary: dict,
    history: list[dict[str, str]],
) -> str:
    """Build the governance chat intent classification prompt."""
    return GOVERNANCE_CHAT_INTENT_PROMPT.format(
        total_entries=investigation_summary.get("total_entries", 0),
        total_flagged=investigation_summary.get("total_flagged", 0),
        findings_by_category=investigation_summary.get("findings_by_category", {}),
        risk_distribution=investigation_summary.get("risk_distribution", {}),
        overall_risk_score=investigation_summary.get("overall_risk_score", 0),
        tool_descriptions=_format_tool_descriptions(tool_registry),
        history=_format_history(history, limit=4),
        user_message=user_message,
    )


def build_governance_chat_synthesis_prompt(
    user_message: str,
    tool_results: list[dict],
    history: list[dict[str, str]],
) -> str:
    """Build the governance chat response-synthesis prompt."""
    return GOVERNANCE_CHAT_SYNTHESIS_PROMPT.format(
        user_message=user_message,
        tool_results=_format_tool_results(tool_results),
        history=_format_history(history, limit=2),
    )
