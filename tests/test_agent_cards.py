from dieta_agentic.dieta_workflow import list_agent_cards

def test_agent_cards_are_complete():
    cards = list_agent_cards()
    ids = {c['agent_id'] for c in cards}
    assert {'triage_agent','candidate_entity_agent','candidate_relation_agent','critic_agent','grounding_agent','feedback_controller'} <= ids
    for c in cards:
        assert c['tools']
        assert c['output_artifacts']
