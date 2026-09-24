from types import SimpleNamespace


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return SimpleNamespace(content=self.responses.pop(0))
