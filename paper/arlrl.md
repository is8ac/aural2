# Learning to understand humans: Exploiting human nature to learn in-band negative reward signals and/or minimizing human frustration when requesting boolean feedback

## Abstract
Knowing what humans want is a differential and important problem.
Traditionally, training such systems requires significant quantities of manually annotated data.
Obtaining this data is difficult.
We describe arlRL, a system which, purely by interacting with a human user via voice, can learn to detect when the user thinks arlRL has taken the wrong action, and, using this information, learns to predict the degree to which an action will displease the user.
ArlRL achieves this with no initial information regarding the detection of emotions in humans, or of the happiness likely to be caused by any actions.
Furthermore, it uses a single model for both classification of reward, and of right action, while avoiding wireheading.


## Introduction
It is generally desirable that machines should take actions which the user likes.
We assume that if the machine takes an wrong action, the user will be displeased regarding the action.
Therefore, the user being displeased regarding an action is evidence that it was not a good action.
If we can measure the displeasure of the user, we can use standard supervised learning to predict it.
As the best action is the action which will cause the least displeasure, we can use such a model to choose the best action.
To be more precise, arlRL avoids taking actions which it predicts will cause the user to be displeased.

Detecting that the user is displeased as a result of an action is nontrivial.

However let us temporally set this problem aside while I describe a further complication.

The goodness of an action is not fixed, but rather changes with the environment state.
There are times when turning on the lights makes the user happy, but once the lights are on, turning on them on again will not make the user any happier.
If the user wishes to sleep, turning on the light even once will cause the user to be unhappy.
When interacting with humans, the correct behavior of a machine is dependent on the state of the environment, particularly the internal state of the human user.
It is expensive and impractical to directly observer a humans internal state.
However any observable information which changes as a function of the internal state of a human can be used to gain information about the humans internal state.
Many bits of information about a humans internal state leek out into the environment.
We can even train the user to expose more bits of information about their internal state.

Examples of easily observable environment state which contain information about the users internal state include facial expression, hand movements, voice, etc.

Given a finite set of actions which the machine can perform, for any given state of the environment, any given action will alter the users happiness by a certain amount.
The task of arlRL then is to learn to predict the degree by which each action will alter the users happiness, thereby allowing us to simple pick the action predicted to be most good.

This then brings us back to the problem of classifying user happiness.
Happiness, like wanting the lights to be on, is just another aspect of a humans internal state.
Information about the users happiness leeks out and is observable.
We must therefore learn to classify the user happiness.

Let us assume that humans know how happy they are.
Image that arlRL detects an state of the world which it thinks is indicative of user unhappiness, and then says to the user "I apologize for the mistake."
If the user was unhappy as a consequence of an action which arlRL had previously taken, the user will be somewhat happier that arlRL at least apologize.
If on the other hand, arlRL had made no mistake, the user will be unhappy that arlRL is pointlessly apologizing.

In other words, if apologizing will make the user happier, arlRL made a mistake previously.
If we treat apologizing as another action whose goodness is predicted just as the goodness of turning the lights on is predicted, we can use it as a proxy for user unhappiness.

If arlRL predicts that taking apologize action will increase user happiness, the user is probably unhappy, and arlRL must have done something wrong.

Consider the inverse.
If arlRL detects a state of the world which it thinks is evidence that the user is happy, and it then says "The correct action was taken".
If a wrong action had previously been taken, the user will be displeased that arlRL is wrong.
But if the correct action was taken, the user may by pleased that arlRL is correct, or they may be displeased that arlRL is wasting there time with obvious statements.
Imagine that the user _did_ consistently express happiness when arlRL correctly said that it had acted correctly.
ArlRL would predict that saying "The correct action was taken", would result in an increases in user happiness, any time the user was happy.
This would create a positive feedback loop of arlRL endlessly stating that its action were good, wire-heading its user.
Fortunately, users are not made happy by machines repeatedly making true statements about the correctness of their past actions.

Therefore, it is only safe / stable to include an action which arlRL will learn to take in response to negative feedback.
ArlRL will learn to take the apologize action when it does make a mistake, but not so strongly as to learn to purposely make mistakes just so as to be rewarded for apologizing.

I now present various analogies.


## Learning to maximize human happiness
Consider a home automation computer.
It can perform various actions, such at turning on or off the lights, or playing music.

When the user expresses there desire for the lights to be turned on, and the lights promptly turn on, the user is happy.
If the lights fail to turn on when the user wants them to, or they turn on at the wrong time, the user is unhappy.

Peoples words often contain strong evidence of their desires.
If a person says "I want the light to be on", or "Please turn on the lights", they probably want the light to be turned on.

If people always expressed their desires in consistent, unambiguous sentences, it would be an easy problem to train a model to recognize the users words and take action based on them.

However people often express both desires and happiness in complex and inconsistent ways.

It is therefore desirable that

People express happiness in various ways.
If a person says "Good machine", they probably approve of the action which the machine has recently taken.
If a person says "Bad machine", they probably dislike the action which the machine has recently taken.


## K armed bandit analogy
Imagine that we have `k` slot machines.
Each slot machine has a probability, initially unknown to us, of giving a reward when its lever is pulled.
These rewards may be negative.
We must pull the lever of one machine every 'd' seconds.
We wish to take the action which our reward.
So far, this a normal multi armed bandit problem.

Now let us add the further complementation that the reward is delivered to us after a varying delay.
Now we wish to take the action which will give us the greatest expected future reward.

Now imagine that the reward probability changes as a function of the current and past environment.
We can observe the environment `e`, here represented as an array of floating point numbers.
Our task then is to predict future reward for each slot machine, for the current environment state.

To make things yet harder, we can't directly observe the reward.
Rewards alter the environment in some way initially unknown to us.
To be able to maximize future reward, we must learn to transform environment states back into rewards.

This may appear to be an imposable task.
Fortunately, there are two special slot machines, whose reward is partial determined by a specific boolean environment value, and we can set this value.
We set the boolean value to our belief about the reward we have received, true if positive, false if negative.
The first machine will look at our belief about the reward as represented in the environment value, and the true reward, and give us a positive reward if our belief is true, and a negative reward if it is false.
The second machine will likewise look at the environment value representing our belief, and the true reward, and give us a positive reward if our belief is false, and a negative reward if it is true.

First machine:

actual reward | agent belief about reward  | final reward
----|------ |-----------
+ | true | +
+ | false | -

Second machine:

actual reward | agent belief about reward  | final reward
------|------ |-----------
- | true | -
- | false | +

### Lookup table
Maintain a table of `k` by number of distinct environment states.
When we receive a reward, look up the cell of the table corresponding to the environment state and machine that caused the reward.
- If the reward is positive, increase the value in the cell.
- If the reward is negative, decrease the value in the cell.

### Strategy
- Every `d` seconds, observe the environment. Look up the environment in our reward table, and pull on the lever of the slot machine whose expected reward is greatest.
- When we observe an environment state which we believe to indicate a positive reward, we should pull the lever of the first of the two special machine.
- When we observe an environment state which we believe to indicate a negative reward, we should pull the lever of the second of the two special machines.

Lookup tables scale badly, so lets replace it with an NN.
Now we have deep q-learning!
