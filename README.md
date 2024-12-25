# Pattern Recognition
**NOVELTY***

What’s your data type? -> Text, sentences (for flexibility)\n

Did we apply optimality? If yes, explain? -> Yes (feature reduction using forward….) we used fuzzy algorithm (rapid fuzz) with semantic algorithm (sentence transformer, word net) and we tried to add several other methods, and each one of them separately. The special one with the highest accuracy is chosen (fuzzy + semantic)\n

Which algorithm did we use? -> fuzzy and semantic\n 
These ones are the ones with the least number of unknown dishes found\n
Fuzzy -> tomato = tomatoes\n
Semantic -> sweet = sugary, hot = spicy\n
KNN, bayesian and other algorithms were tried but incorrect results were found

Match score formula:
Fuzzy score (1)\n
Semantic score (1)\n
fuzzy:0.5\n
semantic:0.3\n
Maximum is taken -> 0.5\n
Threshold -> 0.75 (customized) !optimality!\n
0.5<0.75 -> this is potentially spoiled!\n


0.2 * fuzzy + 0.8 * semantic = score 

Hypothetically
