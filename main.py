import json

# import and setup GPT4All
from gpt4all import GPT4All
import openlit

import os.path

openlit.init(otlp_endpoint="http://127.0.0.1:4318", collect_gpu_stats=True)

model = GPT4All(
    "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",  # downloads / loads a 4.11GB LLM
    device="cuda:NVIDIA GeForce RTX 3060",
)

# Select JSON input
fpInput = open("data/BBG/RankedGames.json", "r", encoding="utf8")
# print(fp.read())
fileContent = fpInput.read()

# read JSON file
data = json.loads(fileContent)
# print(data)

# Loop each game
for game in data:

    # Extract: name, BGGID, publisher(s), categories
    name = str(game["name"])
    BGGID = game["bGGid"]
    slug = game["slug"]
    description = game["description"]
    rating = game["rating"]
    # print("Game " + name + " with BGG ID " + str(BGGID))

    outputFileName = "outputs/" + str(BGGID) + "_" + slug + "_Review.txt"
    if not os.path.isfile(outputFileName):

        with model.chat_session(
            system_prompt="""### System:
You are an AI reviewer of board games. When a human gives you a board game title and its description, you respond with the review.""",
            prompt_template="""### Human:
{0}

### Assistant:
""",
        ):
            review = model.generate(
                "write me a review of the board game "
                + name
                + ". the review must include the following sections: 'overview' this section should be maximum 2 paragraphs. "
                + "'the game at a glance' this section should be a list of game mechanics, genre and suitable age. "
                + "The next section 'game mechanics' should be 2 or 3 paragraphs long with a list of game mechanics at the end "
                + "with an explanation of each mechanic. "
                + "the next section will be 'components and artwork' this section should be 2 or 3 paragraphs long "
                # + "and have a list at the end of the section entitled 'whats in the box' and list out all the games components. "
                + "the next section will be 'replayability and depth'. this section should be 2 or 3 paragraphs long. "
                + "the next section should be called 'accessability and fun factor' this section should be 2 or 3 paragraphs long and have a list at the end "
                + "of the section titled ' who's it for?' and list which sort of people will enjoy the game and its ideal age group. "
                + "a section of 'critisisms'. this section includes at least one negative about the game. "
                + "the final section will be called 'conclusion' and give a roundup of the game. "
                + "Please add a user score of "
                + str(rating)
                + " to the end of the article with a 5 point rundown of notable features of the game. "
                + "Here is a description of the game: "
                + description,
                max_tokens=4096,
            )

        fpOutput = open(
            outputFileName,
            "x",
            encoding="utf8",
        )
        fpOutput.write(review)
        fpOutput.close()
