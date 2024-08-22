import json

# import and setup GPT4All
from gpt4all import GPT4All

model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")  # downloads / loads a 1.98GB LLM

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
    # print("Game " + name + " with BGG ID " + str(BGGID))

    with model.chat_session():
        review = model.generate("Write a review of the board game " + name)

    fpOutput = open("outputs/" + str(BGGID) + "_Review.txt", "x", encoding="utf8")
    fpOutput.write(review)
    fpOutput.close()
