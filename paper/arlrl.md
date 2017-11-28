# Learning what people want by exploiting human nature to learn in-band negative reward signals

## Abstract
At any moment, there are numerous action which a machine may take.
Some of these actions are good, others are bad.
We would like our machines to take good actions and not to take bad ones.
The goodness of an action is not fixed, it is dependent on the state of the environment.
We wish to have a function that takes a environment state and an action and returns an expected reward.

Knowing what action to take humans want is a differential and important problem.
Traditionally, training such systems requires significant quantities of manually annotated data.
Obtaining this data is difficult.
We describe arlRL, a system which, purely by interacting with a human user via voice or other methods, can learn to detect when the user thinks arlRL has taken the wrong action, and, using this information, learns to predict the degree to which an action will displease the user.
ArlRL achieves this with no initial information regarding the detection of emotions in humans, or of the happiness likely to be caused by any actions.
Furthermore, it uses a single model for both classification of reward, and of right action, while avoiding either annoying the user or wireheading itself.


## Introduction
It is generally desirable that machines should take actions which the user likes.
We assume that if the machine takes a wrong action, the user will be displeased regarding the action.
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
User can even intentionally leak more bits about their internal state by talking or other means.

Given a finite set of actions which the machine can perform, for any given state of the environment, any given action will alter the users happiness by a certain amount.
The task of arlRL then is to learn to predict the degree by which each action will alter the users happiness, thereby allowing us to simple pick the action predicted to be most good.

This then brings us back to the problem of classifying user happiness.
Happiness, like wanting the lights to be on, is just another aspect of a humans internal state.
Information about the users happiness leeks out and is observable.
We must therefore learn to classify the users happiness from the information available.

Imagine that arlRL detects a state of the world which it thinks is indicative of user unhappiness, and then says to the user "I apologize for the mistake."
If the user was unhappy as a consequence of the action which arlRL had previously taken, the user will be somewhat happier that arlRL at least apologize.
If on the other hand, arlRL had made no mistake, the user will be unhappy that arlRL is pointlessly apologizing.

In other words, if apologizing will make the user happier, arlRL made a mistake previously.
If we treat `apologize` as another action whose goodness is predicted just as the goodness of turning the lights on is predicted, we can use its goodness as a proxy for user unhappiness.

If arlRL predicts that taking the `apologize` action will increase user happiness, then arlRL must have done something wrong.

Consider the inverse.
Imagine that arlRL detects a state of the world which it thinks is evidence that the user is happy, and it then says "The correct action was taken".
If a wrong action had previously been taken, the user will be displeased that arlRL is wrong.
But if the correct action was taken, the user may by pleased that arlRL is correct, or they may be displeased that arlRL is wasting their time with obvious statements.
Imagine that the user _did_ consistently express happiness when arlRL correctly said that it had acted correctly.
ArlRL would predict that saying "The correct action was taken", would result in an increases in user happiness, any time the user was happy.
This would create a positive feedback loop of arlRL endlessly stating that its action were good, wire-heading its user and/or itself.
Fortunately, users are not made happy by machines repeatedly making true statements about the correctness of their past actions.

Therefore, it is only safe / stable to include an action which arlRL will learn to take in response to _negative_ feedback.
ArlRL will learn to take the apologize action when it does make a mistake, but not so strongly as to learn to purposely make mistakes just so as to be rewarded for apologizing.

I now present various analogies.

## K armed bandit analogy
Imagine that we have a set of slot machines $A$.
Each slot machine has a probability, initially unknown to us, of giving a positive or negative reward $R$ when its lever is pulled.
We must pull the lever of one machine every $d$ seconds.
We wish to take the action which maximizes $R$.
So far, this is a fairly normal multi armed bandit problem.

Now imagine that the reward probability changes as a function of state $s$, where $s$ is a function of current and past environment.
We can observe the environment $e$, and from our current and past observations of $e$, calculate $s$.
$s_t$ is a function of $e_{t}, e_{t-1}, e_{t-2}...$
Our task then is to predict reward for each slot machine, for any given $s$.

To make things yet harder, we can't directly observe $R$.
However, $e$ containes information about $R$, in some way initially unknown to us.
To be able to maximize future reward, we must learn to find the value of $R$ from our observations of $S$.

This may appear to be an imposable task.
Fortunately, there is a special slot machines, $I$, whose reward is the inverse of the most recent reward.
In other words, if the most recent reward was positive, this machine will give us a negative reward, but if the most recent reward was negative, it will give us a positive reward, although not quite as large as inverse of the earlier reward.

### Lookup table
Maintain a table of $|A|$ by $|S|$ cells.
In other other words, for each combination of unique environment state and machine, their exists a distinct cell.
Each time we observe $e$ and compute a new $s$, look up the value of the cell for $a_1$ and the current $S$.

A     | $S_0$ | $S_1$  | $S_2$
------|-------|--------|----
$A_0$ |$-0.1$ | $+0.7$ | $+0.8$
$A_1$ |$+0.8$ | $-0.3$ | $-0.7$
$A_2$ |$-0.2$ | $-0.2$ | $-0.9$
Table: Example value table for three actions and three states.
