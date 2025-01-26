# Efficient Uncertainty Quantification in RL Exploration with Bandit Algorithms
_Preface: This documents my introduction to bandit algorithms in the context of RL. Posting for my clarity, future reference and for others who may benefit._

## Problem: Learning to make efficient decisions

Imagine you're planning your day. Should you grab an umbrella? What's for lunch? What are your goals this year? Decision-making happens at multiple scales - and this happens as well in RL.

In RL, decisions are made at the step- and trajectory- level:
### Stepwise Decisions
A stepwise decision is an agent’s basic decision-making unit: “Given the current situation (s), which action (a) should I take?” Technically, the agent has a policy function $\pi$ that maps states to actions. This level of decision-making focuses on selecting actions in response to immediate observations.

![What Action](https://private-user-images.githubusercontent.com/29778721/406716758-d4340af8-ecf9-4821-b47e-6bb0ab16679c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzc4ODA4MTAsIm5iZiI6MTczNzg4MDUxMCwicGF0aCI6Ii8yOTc3ODcyMS80MDY3MTY3NTgtZDQzNDBhZjgtZWNmOS00ODIxLWI0N2UtNmJiMGFiMTY2NzljLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMTI2VDA4MzUxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ2YzA3ZWYyN2U4OGVhZjIzOWIyMTk2OWU2ZmUzNmFmMWY0NjBkZWNiYzUyMDI2NzIwZjZmZjhlOGQ3Y2YxODImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.AHMSNER5-ROUdmnZW5yNgOTyYHim04Bka-Am0qLf6PQ)
*"What action should I take?"*
### Trajectory Decisions
Beyond single steps, RL cares about entire trajectories. A trajectory is denoted by  
$$\tau = (s_{0}, a_{0}, r_{1}, s_{1}, a_{1}, \dots, s_{T}),$$  
This represents a sequence of states, actions, and rewards. RL aims to optimize the total return across that sequence:  
$$
\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t\,r_{t+1} \right].
$$

• If the environment continues indefinitely, we use a discount factor γ to ensure rewards in the distant future are valued slightly less.  
• If the environment has a clear end—like a board game with win/lose conditions—we call it a finite horizon.

![Trajectory](https://private-user-images.githubusercontent.com/29778721/406716779-76b136ca-f94a-4f1b-8264-7c904ff5b2a0.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzc4ODA4MTAsIm5iZiI6MTczNzg4MDUxMCwicGF0aCI6Ii8yOTc3ODcyMS80MDY3MTY3NzktNzZiMTM2Y2EtZjk0YS00ZjFiLTgyNjQtN2M5MDRmZjViMmEwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMTI2VDA4MzUxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWYxN2VlMjhmNzFlZDA4YzZhZWUwNzk2N2EyYmQwOGYxNmYzYjU5N2JhYzUyMjJlNTZiNTdmZmI0YWM3NmRmYmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.WEW0ExZ7NgH8KjlEnWuSnRIXyidN1NQvsGXr_K5GQms)
*"What is the reward?"*
### Where learning happens in RL
Many modern RL algorithms adopt a policy-based approach, collecting trajectories (or batches of them) and then updating their policy parameters $\theta$ by maximizing an objective function $J(θ)$. 

Compared to earlier methods such as Q-learning or SARSA, which rely heavily on tabular or simpler function approximations, policy-gradient methods let us use powerful neural networks and handle continuous action spaces more easily.
![Learning](https://private-user-images.githubusercontent.com/29778721/406716795-590a799c-1428-4041-b6b3-0e778ea16ca9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzc4ODA4MTAsIm5iZiI6MTczNzg4MDUxMCwicGF0aCI6Ii8yOTc3ODcyMS80MDY3MTY3OTUtNTkwYTc5OWMtMTQyOC00MDQxLWI2YjMtMGU3NzhlYTE2Y2E5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMTI2VDA4MzUxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjYyOWRmMDI5OTBiODczMTU0NjY5N2VjMTFlNWVhODgxZTcwOTVkN2MyMDQ2ZWQ1MmFlYTk3MjlkZDJiNzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0._g64OEm-8a0XDzLBQbi-yLpIIBXIEmmGVQn8zrxWSvg)

In policy-based learning, the policy $\pi_{\theta}$​ is typically updated after collecting trajectories (or batches of trajectories). During data collection, the policy parameters $\theta$ remain fixed. Stepwise optimization within a trajectory does not happen: the action taken simply the outcome of $\theta$. 

Here's an example using PPO: 
```python
# PPO Pseudocode with Structured Blocks  
for iteration in iterations:  
    # [Run Trajectory]  
    # Collect trajectories using current fixed policy π_θ  
    trajectories = rollout(env, policy)  
    
    # [Evaluate Objective Function J(θ)]  
    # Compute advantages and surrogate objective (clipped loss)  
    advantages = compute_advantages(trajectories)  
    surrogate_loss = clipped_surrogate_objective(trajectories, advantages)  
    
    # [Optimize Parameters θ]  
    # Perform gradient ascent on J(θ) = E[surrogate_loss]  
    optimizer.zero_grad()  
    surrogate_loss.backward()  
    optimizer.step()  
    
    # [Update Policy]  
    # Implicit: Policy parameters θ are updated via optimizer.step()  
    # (The "policy" is now π_{θ_new} for the next iteration)  
```

Another with REINFORCE:
```python
# REINFORCE Pseudocode with Structured Blocks  
for episode in num_episodes:  
    # [Run Trajectory]  
    # Collect trajectory using current fixed policy π_θ  
    states, actions, rewards = run_episode(env, policy)  
    
    # [Evaluate Objective Function J(θ)]  
    # Compute Monte Carlo returns: R(τ) = Σ γ^t r_t  
    returns = compute_returns(rewards, gamma=0.99)  
    log_probs = policy(states).log_prob(actions)  
    objective = torch.mean(log_probs * returns)  # J(θ)  
    
    # [Optimize Parameters θ]  
    # Perform gradient ASCENT (maximize J(θ))  
    optimizer.zero_grad()  
    (-objective).backward()  # Minimize -J(θ) ≡ Maximize J(θ)  
    optimizer.step()  
    
    # [Update Policy]  
    # Policy parameters θ are updated via optimizer.step()  
```

Through multiple trajectories, RL algorithms optimize to make good decisions. 

But... how can we make this more efficient? How can an algorithm make "good" decisions at the scale of steps?  
## Motivation: $Good^{tm}$ Learning at the stepwise scale

Why do we need “$Good^{tm}$” decision-making at the stepwise scale? 

In many real-world settings— e.g. medical trials, aerospace testing, or financial investments—each action can be costly or irreversible. We cannot simply replay the same situation over and over to explore all possible choices. Therefore, our algorithms must be highly data-efficient and learn to make good decisions at every step.

For example:
1. Trials are limited (e.g., medical trials or aerospace testing).  
2. Decisions are irreversible (e.g., patient treatments, high-value trading).  
3. Exploration is expensive (e.g., large ad campaigns, robot hardware).  

Here are some illustrations of what $Good$ Decisions can look like:
![Trading](https://private-user-images.githubusercontent.com/29778721/406716802-6d78c0c2-b728-4967-aa7f-0817dfe77b59.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzc4ODA4MTAsIm5iZiI6MTczNzg4MDUxMCwicGF0aCI6Ii8yOTc3ODcyMS80MDY3MTY4MDItNmQ3OGMwYzItYjcyOC00OTY3LWFhN2YtMDgxN2RmZTc3YjU5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMTI2VDA4MzUxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTI1NDM3YjYyYjA0MWVjOTBjYTg5MjY1MTAwNWY2Yjk2ZDNmOWQwNWYxODJjYWEyNzQxYzZlMmQ1NTUzMmRkYzgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j1EhTNxSuu0P3Gb2nTSBod2MpFIznT6aurWDArk-n1U)
_"Trading: Thompson Sampling (Balanced Exploration-Exploitation) vs. Greedy Strategy (Myopic Exploitation)"_  
- _Good Decision_ = Balance exploration (testing new assets) + exploitation (capitalizing on known winners).
- _Bad Decision_ = Blindly chasing short-term gains without probing for better options.

![Trials](https://private-user-images.githubusercontent.com/29778721/406716820-478d407e-bf31-474a-b85c-b478948dd6f4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzc4ODA4MTAsIm5iZiI6MTczNzg4MDUxMCwicGF0aCI6Ii8yOTc3ODcyMS80MDY3MTY4MjAtNDc4ZDQwN2UtYmYzMS00NzRhLWI4NWMtYjQ3ODk0OGRkNmY0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMTI2VDA4MzUxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJiOTBhZjY4MjNhNDlmOTg2MjBmYTFjY2YxMDJiMTRlYTNiN2ExOWVlZWI1ZGI0ZTVjMzFhNzhiMTU2MGU3ZGMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.dCfqbnEB23z_pionHORMv252DQZ0-zDgAqKq4wclT5Q)
_"Medical Trials: Conservative UCB (Risk-Averse Prioritization) vs. Standard UCB (Overconfident Exploration)"_
- _Good Decision_ = Prioritize patient safety by constraining exploration within ethical bounds.
- _Bad Decision_ = Ignoring uncertainty and deploying high-risk treatments prematurely.

In all these scenarios, if an algorithm acts sub-optimally, we cannot “Ctrl+Z” and try again. There are consequences. Hence, the question becomes: how can we efficiently learn from sparse data, balancing risk, exploration, and exploitation?

## Solution: Bandits for Step-level Decision Making
Bandit algorithms are a great domain to focus on this question. 

In fact, I think its useful to look at bandit algorithms as on the spectrum of RL. Here are some similarities between bandits and RL:  

| **Aspect**              | **Non-Contextual Bandit** | **Contextual Bandit**           | **Reinforcement Learning (RL)**          |
| ----------------------- | ------------------------- | ------------------------------- | ---------------------------------------- |
| **State/Context**       | None                      | Exogenous $s_t$ (IID)           | Endogenous $s_t$ (depends on $a_{<t}$)   |
| **Policy**              | $\pi(a)$                  | $\pi(a \mid s_t)$               | $\pi(a \mid s_t)$                        |
| **State Transitions**   | N/A                       | $s_{t+1} \perp a_t$ (no effect) | $s_{t+1} \sim P(\cdot \mid s_t, a_t)$    |
| **Objective**           | $\max \mathbb{E}[r_t]$    | $\max \mathbb{E}[r_t \mid s_t]$ | $\max \mathbb{E}[\sum \gamma^k r_{t+k}]$ |
| **Temporal Dependence** | None (single-step)        | None (single-step)              | Full (multi-step)                        |
| **Example**             | Slot machines             | Personalized ads                | Game-playing AI, self-driving cars       |

Though bandits do not handle time-varying states or long-term returns, they focus on exactly the same challenge: taking the best next action while accounting for uncertainty. 

They do this by answering two critical questions:
1. "What's the best action right now?"
2. "Which action might surprise us if we try it?"

Here is a pseudo-code:
```python
# ACTION SELECTION
def select_action(
	actions: collection of actions,
	context: vector of context
):
	actions = []
	For each action in actions:
		action_score = score_reward(action) + score_uncertainty(action)
		actions.append(action_score)
	return argmax(action)

# ACTION UPDATE
def update_action(
	action_taken: key to action taken,
	reward: reward from action,
	context: vector of context (e.g. age, etc)
):
	for actions[action]:
		update_reward_matrix(action, reward)
		update_uncertainty_matrix(action, optional: context) # This can be a simple count or a context vector
```

You will observe:
- **`choose_action`**: Prioritizes actions with high rewards _or_ high uncertainty (never tried).
- **`update_action`**: Lightweight update—no gradients, just counts and averages.

By focusing on bandit problems, we are focusing on learning at the step-wise scale by quantifying uncertainty.
## Approaches: Bandit Approaches to Uncertainty Quantification

To tackle uncertainty systematically, we look at three main categories of bandit algorithms:

| Approach    | Description                                                                          | Example Analogy                                                                           | Algorithm Example | Rule                                                                                          | Goal                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Frequentist** | Uses confidence bounds derived from observed data.                                    | “We tried Action A ten times: six successes, so we’re ~95% sure the success rate is between 40% and 80%.” | UCB               | Pick actions that have high upper confidence bounds (reward estimate + uncertainty term).    | Prioritize arms that are either high-reward or not well explored, ensuring balanced exploration and exploitation. |
| **Bayesian**    | Maintains a posterior distribution over rewards that updates as more data arrives. | “We started with a 50% guess for Action A’s success rate. After ten tries, we now believe it’s 60%, but it may still surprise us.” | Thompson Sampling | Randomly sample from each arm’s posterior, then pick the arm that samples the highest value. | Naturally balance exploration (by occasionally drawing from uncertain arms) with exploitation (choosing high-mean arms).                           |
| **Heuristic**   | Operates with a simpler, more formulaic approach.                                  | N/A                                                                                       | ε-Greedy          | Exploit the best arm with probability 1 – ε and explore randomly with probability ε.         | Easy to implement and baseline-friendly, though sometimes less efficient than the other approaches.        |

Each of these methods outlines an approach for dealing with uncertainty at the stepwise scale. More advanced variations exist within each category, but the overarching theme isthe same: bandit solutions highlight how to make the most out of limited data for single-step gains.

Connecting back to RL, this is the exciting thought - can bandit approaches can be treated blueprints for stepwise decision-making in RL? 

By including uncertainty into decision making, its possible that agents are able to learn faster, even in data-scarce worlds (....which is most of meat-space with consequences).

## Wrapup & Motivating Inspiration

Ok, I am now at the edge my knowledge comfort, so I'm going to round this up. I've touched on two key ideas in this post. 
### 1. Policy-Based Methods: Powerful but Sample-Inefficient
RL methods that operate over full trajectories (like policy-gradient algorithms) are exceptionally powerful for complex, sequential tasks. However, they often need large amounts of data to learn robust policies. For real-world applications—such as healthcare or finance—acquiring so many trajectories can be prohibitively expensive or risky.
### 2. Bandit Algorithms: Elegant but Shortsighted
By contrast, bandit algorithms excel at making efficient decisions in single-step scenarios. They explicitly quantify uncertainty (e.g., through confidence bounds or posterior distributions) and balance exploration with exploitation on a step-by-step basis. Yet, they do not consider long-term effects of actions, since there are no state transitions or discounted rewards.
### Putting It All Together: Toward Hybrid Approaches
 “What if we could bring the best of both worlds into a single framework?” Specifically, how might we incorporate bandit-like exploration strategies within each RL timestep?

In theory, there could be hybrid or hierarchical models:
- Uncertainty Aware Value Estimation 
	- Instead of just having a single estimate of the value of taking an action _a_ in state _s_, we maintain a _distribution_ over these values. This distribution represents its uncertainty.
	- $Q(s,a) \rightarrow P(Q|s,a)$
- Thompson Sampling for RL
	- When choosing an action, sample from this distribution, effectively biasing action selection towards actions with high estimated values _and_ high uncertainty. 
	- $Q(s,a) \sim P(Q|s,a),\space choose a = argmax \space Q(s,a)$
- Trajectory Conscious Bandits
	- Extend bandits to incorporate delayed rewards by including discounted returns in the uncertainty estimates
	- ?? 

 To end on a high note, here's a motivating use case of the above ideas, courtesy of AI:

*--- Start AI contribution ---*

**Applied Ideas: Hierarchical Policies in Trading (AI generated)**

Trajectory-level learning ensures agents play the long game, but what about the quality of each step? Hierarchical RL offers a solution: decompose policies into a macro-trajectory planner (RL) and micro-step actor (bandit).  

To ground these ideas, consider a **hierarchical RL** framework for algorithmic trading:
- The macro planner (RL) learns to maximize portfolio returns over quarters, defining high-level goals like “reduce risk exposure.”  
- The micro actor (bandit) handles daily trades, using UCB or Thompson sampling to execute actions (e.g., buy/sell) under the macro policy’s constraints.
- 
This hierarchy mirrors human decision-making: you set yearly goals (trajectories) but make daily choices (steps) adaptively.

`Policy_theta(a|s) = Policy_RL(z|s) * Policy_Bandit(a|s,z)`

Here, $z$ represents latent goals (e.g., “explore energy stocks”), and the bandit policy optimizes immediate actions under $z$. By explicitly modelling both timescales, we retain RL’s strategic depth while injecting bandit-like efficiency at the step level.  

*--- End AI Contribution ---* 

...heady stuff ... more on this next time!



Signing off, 

Lennard Ong

吾之識界，吾之世界

*Wú zhī shí jiè, wú zhī shìjiè*

---

## References
There's no way I learned the above from nothing. Here are the great resources I benefitted from, sharing here to support the curious learner:
- Lectures: [Emma Brunskill - Fast RL 2](https://youtu.be/jJ7JbQBTChM?si=jaJrM1kQPJLcnGuF) 
- Check your learning: [CS234 Coursework](https://web.stanford.edu/class/cs234/)
- Thompson Sampling Paper:  [Thompson Sampling for Contextual Bandits with Linear Payoffs](https://arxiv.org/abs/1209.3352)
- UCB Paper: [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146)
- for AI learning support, `deepseek` has been an amazing help.
