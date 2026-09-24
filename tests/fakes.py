from types import SimpleNamespace


class FakeLLM:
    def __init__(self, responses, usage=False):
        self.responses = list(responses)
        self.usage = usage
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        resp = SimpleNamespace(content=self.responses.pop(0))
        if self.usage:
            from graph.budget import count_tokens
            resp.usage_metadata = {"total_tokens": count_tokens(prompt) + 150}
        return resp
