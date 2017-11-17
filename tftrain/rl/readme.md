# Aural reinforcement learning

Two models:
- Causality assignment LSTM. Runs asynchronously backwards in time, identifying the utterances whose interpretation by the current model, the user will express displeasure regarding. Trained via normal supervised learning.
- Intent LSTM. Runs forewords, classifying intents, one of which is the mistake intent.

Intent LSTM running forewords in real time classifies intents.
When intent is detected, action is triggered.
If human expresses feedback regarding the action, the causality assignment LSTM is triggered, starting with the end of the feedback, and running backwards in time for some fixed duration.
The causality assignment LSTM starts with the user feedback, positive or negative, and runs backwards through time looking for the utterance which was misinterpreted.
Once the utterance which is wrongly classified by the current intent LSTM is identified, the loss is back propagated across the weights.
