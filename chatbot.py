"""
OptiScholar Chatbot — Rule-Based Intent Classifier
====================================================
Handles 6 intents via keyword matching + predefined responses.
Integrates with the existing recommendation pipeline.

Intents:
  1. greet          — hello, hi, start
  2. find_scholarships — find, recommend, show me scholarships
  3. eligibility_gap  — gap, missing, what do I need, improve
  4. scholarship_info — tell me about, explain, what is [scholarship]
  5. general_advice   — tips, advice, how to apply, increase chances
  6. fallback         — anything not matched

Design: stateful conversation — remembers student profile and
last recommendations across turns.
"""

import re
import random
from typing import Optional


# ══════════════════════════════════════════════════════════════
# SECTION 1 — INTENT PATTERNS
# ══════════════════════════════════════════════════════════════

INTENTS = {
    "greet": [
        r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bstart\b",
        r"\bgood morning\b", r"\bgood afternoon\b", r"\bhelp me\b",
        r"\bwhat can you do\b", r"\bwho are you\b"
    ],
    "find_scholarships": [
        r"\bfind\b.*scholarship", r"\brecommend\b",
        r"\bshow me\b", r"\blist\b.*scholarship",
        r"\bwhat scholarship", r"\bwhich scholarship",
        r"\bget scholarship", r"\bsearch\b",
        r"\bsuitable\b", r"\bmatching\b", r"\bfor me\b"
    ],
    "eligibility_gap": [
        r"\bgap\b", r"\bmissing\b", r"\bnot eligible\b",
        r"\bwhat do i need\b", r"\bimprove\b", r"\bclose to\b",
        r"\balmost\b", r"\bnear\b.*qualify", r"\bhow can i qualify\b",
        r"\bwhat.*required\b", r"\broadmap\b", r"\baction\b"
    ],
    "scholarship_info": [
        r"\btell me about\b", r"\bexplain\b", r"\bwhat is\b",
        r"\bdetails\b", r"\bmore info\b", r"\bdescription\b",
        r"\bhow much\b", r"\bdeadline\b", r"\brequirement\b",
        r"\beligib\b", r"\bwho can apply\b", r"\bamount\b"
    ],
    "general_advice": [
        r"\btips\b", r"\badvice\b", r"\bhow to apply\b",
        r"\bincrease\b.*chance", r"\bbetter chance\b",
        r"\bprepare\b", r"\bapplication\b", r"\bessay\b",
        r"\bgpa\b.*improve", r"\braise\b.*gpa",
        r"\bfinancial\b.*aid", r"\bstrategy\b"
    ],
    "profile_update": [
        r"\bmy gpa\b", r"\bmy degree\b", r"\bi am\b.*student",
        r"\bupdate\b.*profile", r"\bchange\b.*gpa",
        r"\bi have\b.*gpa", r"\bmy income\b"
    ]
}


# ══════════════════════════════════════════════════════════════
# SECTION 2 — RESPONSE TEMPLATES
# ══════════════════════════════════════════════════════════════

GREET_RESPONSES = [
    "👋 Hello! I'm OptiScholar, your AI scholarship assistant.\n\n"
    "I can help you:\n"
    "• 🎓 Find scholarships matching your profile\n"
    "• 📋 Explain your eligibility gaps\n"
    "• ℹ️ Answer questions about specific scholarships\n"
    "• 💡 Give you advice on improving your application\n\n"
    "To get started, tell me a bit about yourself or ask me to find scholarships for you!",

    "Hi there! 👋 Welcome to OptiScholar.\n\n"
    "I'm here to help you find the right scholarships and guide you through the application process.\n\n"
    "What would you like to do?\n"
    "• Find scholarships for my profile\n"
    "• Check my eligibility gaps\n"
    "• Get application advice",
]

GENERAL_ADVICE_RESPONSES = [
    """💡 **Scholarship Application Tips:**

**1. Improve your GPA**
Even a small GPA improvement (e.g. 2.8 → 3.0) can open up Academic Excellence scholarships. Focus on your next semester's performance.

**2. Document financial need early**
For Need-Based scholarships, gather income statements and bank records. Many scholarships require official documentation.

**3. Apply to multiple types**
Don't focus only on Merit-Based. Community Service and Athletic scholarships often have lower competition.

**4. Meet deadlines**
Most scholarships have fixed deadlines. Set calendar reminders 2 weeks before each deadline.

**5. Write a strong personal statement**
Focus on your specific goals and how the scholarship helps you achieve them. Be specific, not generic.

**6. Check eligibility filters carefully**
Many rejections happen because applicants miss a citizenship or enrollment requirement. Read the full description.

Would you like specific advice for your profile?""",

    """📚 **How to Maximize Your Scholarship Chances:**

**Check your eligibility carefully** — the eligibility filter in OptiScholar narrows 11,289 scholarships to those you actually qualify for. Start there.

**Use the Gap Finder** — it shows you scholarships you're *almost* eligible for. A 0.2 GPA improvement could unlock several new options.

**Prioritize by amount and competition** — smaller scholarships (under $1,000) have lower competition. Apply to several rather than only targeting large awards.

**Community Service scholarships** have the highest approval rates in our dataset (58.6%). If you have any volunteer experience, highlight it.

Want me to find the best scholarships for your current profile?"""
]

FALLBACK_RESPONSES = [
    "I'm not sure I understood that. Could you rephrase?\n\n"
    "I can help you with:\n"
    "• **Finding scholarships** — 'Find scholarships for me'\n"
    "• **Eligibility gaps** — 'What am I missing to qualify?'\n"
    "• **Scholarship details** — 'Tell me about [scholarship name]'\n"
    "• **Application advice** — 'How do I improve my chances?'",

    "Hmm, I didn't quite get that. 🤔\n\n"
    "Try asking me something like:\n"
    "• 'What scholarships match my profile?'\n"
    "• 'Show me my eligibility gaps'\n"
    "• 'Give me application tips'",
]


# ══════════════════════════════════════════════════════════════
# SECTION 3 — INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════

def classify_intent(message: str) -> str:
    """Classify user message into one of the defined intents."""
    msg = message.lower().strip()

    # Score each intent by number of pattern matches
    scores = {}
    for intent, patterns in INTENTS.items():
        score = sum(1 for p in patterns if re.search(p, msg))
        if score > 0:
            scores[intent] = score

    if not scores:
        return "fallback"

    # Return highest scoring intent
    return max(scores, key=scores.get)


def extract_scholarship_name(message: str,
                              recommendations=None) -> Optional[str]:
    """
    Try to extract a scholarship name from the message.
    Checks against current recommendations if available.
    """
    if recommendations is None or len(recommendations) == 0:
        return None

    msg_lower = message.lower()

    # Check if any recommended scholarship title appears in message
    for _, row in recommendations.iterrows():
        title = str(row.get("scholarship_title", "")).lower()
        # Check if significant words from title appear in message
        title_words = [w for w in title.split() if len(w) > 4]
        if any(w in msg_lower for w in title_words[:3]):
            return row.get("scholarship_title", "")

    # Check for numbered reference ("tell me about number 3")
    m = re.search(r'\b(number|#|no\.?)\s*(\d+)\b', msg_lower)
    if not m:
        m = re.search(r'\b(\d+)(st|nd|rd|th)?\b', msg_lower)
    if m:
        try:
            idx = int(m.group(2) if m.group(1) in
                      ["number", "#", "no", "no."] else m.group(1)) - 1
            if 0 <= idx < len(recommendations):
                return recommendations.iloc[idx]["scholarship_title"]
        except (ValueError, IndexError):
            pass

    return None


# ══════════════════════════════════════════════════════════════
# SECTION 4 — RESPONSE GENERATORS
# ══════════════════════════════════════════════════════════════

def respond_find_scholarships(profile: dict,
                               recommendations=None) -> str:
    """Generate response for scholarship finding intent."""
    if profile is None or not profile.get("degree_level"):
        return (
            "I'd love to find scholarships for you! 🎓\n\n"
            "First, I need to know a bit about you. Please fill in your "
            "profile in the sidebar, or upload your transcript/CV and I'll "
            "extract your details automatically.\n\n"
            "Key info needed:\n"
            "• GPA\n• Degree level (bachelor/master/PhD)\n"
            "• Financial need\n• Nationality"
        )

    if recommendations is None or len(recommendations) == 0:
        gpa    = profile.get("final_gpa", "unknown")
        degree = profile.get("degree_level", "bachelor")
        return (
            f"Based on your profile (GPA: {gpa}, Level: {degree}), "
            "I'm searching for matching scholarships...\n\n"
            "Please click **'Find My Scholarships'** in the main panel to "
            "run the full recommendation engine. I'll then be able to "
            "explain and discuss the results with you here! 🎯"
        )

    # Recommendations exist — summarize top results
    top_5 = recommendations.head(5)
    gpa   = profile.get("final_gpa", "N/A")
    level = profile.get("degree_level", "bachelor")

    lines = [
        f"✅ Great news! I found **{len(recommendations):,} eligible scholarships** "
        f"for your profile (GPA: {gpa}, Level: {level}).\n",
        "**🏆 Your Top 5 Matches:**\n"
    ]

    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        title  = str(row.get("scholarship_title", ""))[:55]
        stype  = row.get("scholarship_type", "")
        amount = row.get("funding_amount_raw", 0) or 0
        score  = row.get("transfer_score",
                 row.get("bilstm_score",
                 row.get("ncf_score", 0))) or 0
        lines.append(
            f"{i}. **{title}**\n"
            f"   Type: {stype} | Amount: ${amount:,.0f} | Match: {score:.0%}"
        )

    lines.append(
        "\n💬 Ask me about any of these scholarships, or say "
        "'show my eligibility gaps' to see what you're close to qualifying for!"
    )
    return "\n".join(lines)


def respond_eligibility_gap(profile: dict, gap_results=None) -> str:
    """Generate response for eligibility gap intent."""
    if profile is None or not profile.get("final_gpa"):
        return (
            "To find your eligibility gaps, I need your profile first. 📋\n\n"
            "Please fill in your GPA, degree level, and financial need "
            "in the sidebar. Then I can show you which scholarships you're "
            "close to qualifying for and exactly what you need to do!"
        )

    if gap_results is None or len(gap_results) == 0:
        gpa = profile.get("final_gpa", 0)
        return (
            f"Based on your GPA of {gpa:.2f}, let me check your eligibility gaps.\n\n"
            "Please click **'Show Eligibility Roadmap'** in the main panel to "
            "run the Gap Finder. I'll then explain the results and help you "
            "prioritize which gaps to close first! 🗺️"
        )

    # Gap results exist
    lines = [
        f"🗺️ **Your Eligibility Roadmap**\n",
        f"I found **{len(gap_results)} scholarships** you're close to qualifying for!\n"
    ]

    for i, (_, row) in enumerate(gap_results.iterrows(), 1):
        title   = str(row.get("scholarship_title", ""))[:50]
        gap     = row.get("gap_score", 0)
        actions = row.get("action_items", row.get("gaps", ""))
        amount  = row.get("funding_amount", 0) or 0
        lines.append(
            f"\n**{i}. {title}**\n"
            f"   💰 ${amount:,.0f} | Gap score: {gap:.2f}\n"
            f"   ✅ Action: {actions}"
        )

    lines.append(
        "\n\n💡 **Priority advice:** Focus on the lowest gap score first — "
        "that's the scholarship you're closest to qualifying for. "
        "A small GPA improvement often unlocks multiple scholarships at once!"
    )
    return "\n".join(lines)


def respond_scholarship_info(message: str,
                              recommendations=None,
                              opto_df=None) -> str:
    """Generate response about a specific scholarship."""
    sch_name = extract_scholarship_name(message, recommendations)

    if sch_name is None:
        if recommendations is not None and len(recommendations) > 0:
            titles = [f"{i+1}. {str(r['scholarship_title'])[:50]}"
                      for i, (_, r) in enumerate(recommendations.head(5).iterrows())]
            return (
                "Which scholarship would you like to know about? 🔍\n\n"
                "Your current recommendations:\n" +
                "\n".join(titles) +
                "\n\nJust say the number or part of the name!"
            )
        return (
            "Which scholarship are you asking about? 🔍\n\n"
            "Please find scholarships first by clicking "
            "'Find My Scholarships', then I can give you details about "
            "any of the results!"
        )

    # Find scholarship in data
    if opto_df is not None:
        matches = opto_df[
            opto_df["scholarship_title"].str.lower().str.contains(
                sch_name.lower()[:20], na=False
            )
        ]
        if len(matches) > 0:
            row = matches.iloc[0]
            desc    = str(row.get("description_cleaned", ""))
            amount  = row.get("funding_amount_raw", 0) or 0
            stype   = row.get("scholarship_type", "")
            gpa_req = row.get("min_gpa_required", 0) or 0
            need    = row.get("requires_financial_need", 0)
            cit     = row.get("citizenship_required", "")

            desc_short = desc[:300] + "..." if len(desc) > 300 else desc

            return (
                f"📋 **{row['scholarship_title']}**\n\n"
                f"**Type:** {stype}\n"
                f"**Amount:** ${amount:,.0f}\n"
                f"**Min GPA:** {gpa_req/5:.1f}/4.0\n"
                f"**Financial Need:** {'Required' if need else 'Not required'}\n"
                f"**Citizenship:** {cit if cit else 'No restriction'}\n\n"
                f"**Description:**\n{desc_short}"
            )

    return (
        f"I found **{sch_name}** in your recommendations! 📋\n\n"
        "For full details including description, requirements, and deadline, "
        "click on the scholarship in the main panel. "
        "I can help you understand any specific requirement — just ask!"
    )


def respond_general_advice(profile: dict = None) -> str:
    """Generate general scholarship application advice."""
    base = random.choice(GENERAL_ADVICE_RESPONSES)

    if profile and profile.get("final_gpa"):
        gpa = float(profile["final_gpa"])
        if gpa < 2.5:
            base += (
                "\n\n⚠️ **Personal note for your profile:** "
                f"Your GPA of {gpa:.2f} currently limits your Merit-Based options. "
                "Focus on **Need-Based** and **Community Service** scholarships "
                "in the meantime while working on your GPA."
            )
        elif gpa >= 3.5:
            base += (
                "\n\n⭐ **Personal note for your profile:** "
                f"Your GPA of {gpa:.2f} is excellent! You qualify for "
                "**Academic Excellence** scholarships — these often have "
                "lower competition despite higher requirements."
            )
        elif profile.get("financial_need"):
            base += (
                "\n\n💰 **Personal note for your profile:** "
                "Since you have financial need, make sure to complete your "
                "Need-Based scholarship applications first — these are "
                "specifically designed to help students in your situation."
            )

    return base


def respond_profile_update(message: str, profile: dict) -> str:
    """Handle profile update requests."""
    msg = message.lower()

    # Try to extract GPA from message
    m = re.search(r'\bgpa\b.*?(\d+\.?\d*)', msg)
    if not m:
        m = re.search(r'(\d+\.?\d*)\b.*?\bgpa\b', msg)

    if m:
        try:
            new_gpa = float(m.group(1))
            if 0.0 <= new_gpa <= 4.0:
                return (
                    f"Got it! I've noted your GPA as **{new_gpa:.2f}**. 📝\n\n"
                    "Please update this in the profile form on the sidebar "
                    "to rerun the recommendations with your new GPA. "
                    "Would you like me to find scholarships based on this?"
                )
        except ValueError:
            pass

    return (
        "I can see you want to update your profile! 📝\n\n"
        "Please use the **sidebar form** to update your details — "
        "I'll automatically recalculate your recommendations once you save. "
        "What would you like to change?"
    )


# ══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN CHATBOT CLASS
# ══════════════════════════════════════════════════════════════

class OptiScholarChatbot:
    """
    Stateful rule-based chatbot for OptiScholar.
    Maintains conversation history and context across turns.
    """

    def __init__(self):
        self.history         = []   # list of (role, message) tuples
        self.profile         = None
        self.recommendations = None
        self.gap_results     = None
        self.opto_df         = None
        self.turn_count      = 0

    def set_context(self, profile=None, recommendations=None,
                     gap_results=None, opto_df=None):
        """Update chatbot context from the main app."""
        if profile is not None:
            self.profile = profile
        if recommendations is not None:
            self.recommendations = recommendations
        if gap_results is not None:
            self.gap_results = gap_results
        if opto_df is not None:
            self.opto_df = opto_df

    def respond(self, user_message: str) -> str:
        """
        Process user message and return response.
        Main entry point for the chatbot.
        """
        self.turn_count += 1
        self.history.append(("user", user_message))

        intent   = classify_intent(user_message)
        response = self._generate_response(intent, user_message)

        self.history.append(("assistant", response))
        return response

    def _generate_response(self, intent: str,
                             message: str) -> str:
        """Route to the appropriate response generator."""

        if intent == "greet":
            return random.choice(GREET_RESPONSES)

        elif intent == "find_scholarships":
            return respond_find_scholarships(
                self.profile, self.recommendations
            )

        elif intent == "eligibility_gap":
            return respond_eligibility_gap(
                self.profile, self.gap_results
            )

        elif intent == "scholarship_info":
            return respond_scholarship_info(
                message, self.recommendations, self.opto_df
            )

        elif intent == "general_advice":
            return respond_general_advice(self.profile)

        elif intent == "profile_update":
            return respond_profile_update(message, self.profile or {})

        else:
            return random.choice(FALLBACK_RESPONSES)

    def get_history(self) -> list:
        return self.history

    def clear_history(self):
        self.history    = []
        self.turn_count = 0

    def get_suggested_questions(self) -> list:
        """Return context-aware suggested questions."""
        if self.profile is None:
            return [
                "Hello! What can you do?",
                "How do I get started?",
                "What information do you need from me?",
            ]
        if self.recommendations is None:
            return [
                "Find scholarships for me",
                "Give me application tips",
                "What scholarships are available?",
            ]
        return [
            "Show my eligibility gaps",
            "Tell me about the top scholarship",
            "How can I improve my chances?",
            "What do I need to qualify for more scholarships?",
        ]


# ══════════════════════════════════════════════════════════════
# SECTION 6 — STREAMLIT INTEGRATION
# ══════════════════════════════════════════════════════════════

def render_chatbot_ui(chatbot: OptiScholarChatbot):
    """
    Render the chatbot UI in Streamlit.
    Call this from your main Streamlit app.

    Usage:
        from chatbot import OptiScholarChatbot, render_chatbot_ui
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = OptiScholarChatbot()
        render_chatbot_ui(st.session_state.chatbot)
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed: pip install streamlit")
        return

    st.markdown("### 💬 OptiScholar Assistant")
    st.caption("Ask me about scholarships, eligibility, or application tips")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not chatbot.history:
            # Show welcome message
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(random.choice(GREET_RESPONSES))
        else:
            for role, msg in chatbot.history:
                avatar = "🎓" if role == "assistant" else "👤"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(msg)

    # Suggested questions
    if len(chatbot.history) < 2:
        st.markdown("**Quick questions:**")
        suggestions = chatbot.get_suggested_questions()
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions[:3]):
            with col:
                if st.button(suggestion, use_container_width=True,
                             key=f"suggest_{suggestion[:20]}"):
                    response = chatbot.respond(suggestion)
                    st.rerun()

    # Input box
    if prompt := st.chat_input("Ask me anything about scholarships..."):
        response = chatbot.respond(prompt)
        st.rerun()

    # Clear button
    if chatbot.history:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            chatbot.clear_history()
            st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN — STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("OptiScholar Chatbot — Terminal Test Mode")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    bot = OptiScholarChatbot()

    # Set dummy profile for testing
    bot.set_context(profile={
        "final_gpa": 3.2,
        "degree_level": "bachelor",
        "financial_need": 0,
        "International": 0,
        "age": 21,
        "gender": "Female",
        "ses_category": "Middle",
        "household_income": 45000,
    })

    # Show greeting
    print(f"Bot: {bot.respond('hello')}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        intent   = classify_intent(user_input)
        response = bot.respond(user_input)
        print(f"\n[Intent: {intent}]")
        print(f"Bot: {response}\n")
