# Pattern Recognition
**NOVELTY***

What’s your data type? -> Text, sentences (for flexibility)

Did we apply optimality? If yes, explain? -> Yes (feature reduction using forward….) we used fuzzy algorithm (rapid fuzz) with semantic algorithm (sentence transformer, word net) and we tried to add several other methods, and each one of them separately. The special one with the highest accuracy is chosen (fuzzy + semantic)

Which algorithm did we use? -> fuzzy and semantic
These ones are the ones with the least number of unknown dishes found
Fuzzy -> tomato = tomatoes
Semantic -> sweet = sugary, hot = spicy
KNN, bayesian and other algorithms were tried but incorrect results were found

Match score formula:
Fuzzy score (1)
Semantic score (1)
fuzzy:0.5
semantic:0.3
Maximum is taken -> 0.5
Threshold -> 0.75 (customized) !optimality!
0.5<0.75 -> this is potentially spoiled!


0.2 * fuzzy + 0.8 * semantic = score 

Hypothetically
