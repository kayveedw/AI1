import json

# import and setup GPT4All
from gpt4all import GPT4All
import openlit

openlit.init(otlp_endpoint="http://127.0.0.1:4318")
# openlit.init(collect_gpu_stats=True)

model = GPT4All(
    "orca-mini-3b-gguf2-q4_0.gguf",  # downloads / loads a 1.98GB LLM
    device="cuda:NVIDIA GeForce RTX 3060",
)

# Select JSON input
fpInput = open("inputs/400.json", "r", encoding="utf8")
# print(fp.read())
fileContent = fpInput.read()

# read JSON file
data = json.loads(fileContent)
# print(data)

# Loop each game
for game in data:

    # Extract: name, BGGID, publisher(s), categories
    name = str(game["name"])
    BGGID = game["id"]
    description = game["description"]
    # print("Game " + name + " with BGG ID " + str(BGGID))

    with model.chat_session():
        review = model.generate(
            "write me a review of the board game "
            + name
            + ". the review must include the following sections: 'overview' this section should be maximum 2 paragraphs. 'the game at a glance' this section should be a list of game mechanics, genre and suitable age. The next section 'game mechanics' should be 2 or 3 paragraphs long with a list of game mechanics at the end with an explanation of each mechanic. the next section will be 'components and artwork' this section should be 2 or 3 paragraphs long and have a list at the end of the section entitled 'whats in the box' and list out all the games components. the next section will be 'replayability and depth'. this section should be 2 or 3 paragraphs long. the next section should be called 'accessability and fun factor' this section should be 2 or 3 paragraphs long and have a list at the end of the section titled ' who's it for?' and list which sort of people will enjoy the game and its ideal age group. the final section will be called 'conclusion' and give a roundup of the game.Please add a user score out of 10 to the end of the article with a 5 point rundown of notable features of the game. "
            + ". Here is a description of the game: "
            + description,
            max_tokens=4096,
        )

    fpOutput = open("outputs/" + str(BGGID) + "_Review.txt", "x", encoding="utf8")
    fpOutput.write(review)
    fpOutput.close()
