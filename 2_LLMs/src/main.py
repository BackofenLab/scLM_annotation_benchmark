from ollama import chat
from ollama import ChatResponse

from omegaconf.omegaconf import OmegaConf


cfg = OmegaConf.load("../config/llms.yaml")


def get_responses():
    llm = "{}:{}".format(cfg.model_name, cfg.model_version)
    response: ChatResponse = chat(model=llm, messages=[
      {
        'role': 'user',
        'content': 'Does perturbing CCND3 gene in K562 cells cause the gene expression of \
            SDE2 to upregulate or downregulate? Provide response either as Yes or No',
      },
    ])
    print(response.message.content)


if __name__ == "__main__":
    get_responses()
