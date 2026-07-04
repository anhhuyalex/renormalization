Assumption 1: We have a list of $K$ sequences, where each sequence is a list of $N$ bits (either 0 or 1). The bits are independently and identically sampled from the uniform distribution over bits.

Assumption 2: The list is ordered arbitrarily into a series of ranks.

Assumption 3: We have an agent that is presented with a series of items from the list. The probability of the sequence being presented is inversely proportional to the rank of the sequence.

In equations, $P(k) = \frac{1}{Zk}$ where $k$ is the rank of the sequence and $Z(K)$ is the normalizing constant.

Assumption 4: The agent is trying to learn the list by minimizing its average loss across all $N$ bits.

Assumption 5: The agent uses an autoregressive architecture to learn the list.

Assumption 6: Once an agent has learned a sequence in the list, the probability that it recalls a bit at position $j+1$ correctly is $\frac{1}{f_j}$,

where $f_j$ is the number of sequences up until the $j$th bit that are identical to the sequence. It is $\frac{1}{f_j}$ because the agent is still uncertain about the identity of the sequence at position $j+1$, and this uncertainty is quantified by $f_j$. If $f_j$ is 1 (i.e. there is only 1 item in the list up until bit j) then the agent is clearly certain about the sequence at position $j+1$, and all positions after that.

Consider the probability that the model correctly predicts the $j+1$th bit of sequence $s^{(k)}$, given the prefix $s^{(k)}_{0..j}$.

Let us denote this probability by $p(s^{(k)}_{j+1}|s^{(k)}_{0..j})$.

Then, $p(s^{(k)}_{j+1}|s^{(k)}_{0..j}) = \sum_{i=1}^{K} p(s^{(i)}_{j+1}|i)P(i|s^{(k)}_{0..j})$

where $P(i|s^{(k)}_{0..j})$ is the agent's posterior probability of sequence $i$ is being presented, given the prefix $s^{(k)}_{0..j}$
and where $p(s^{(i)}_{j+1}|i)$ is the probability of the $j+1$th bit being correct given sequence $i$.
This probability is known as the posterior predictive probability. The interpretation of this is that the agent is summing over all possible sequences that could have been presented to the agent, 
and then computing the posterior probability of the sequence $i$ being presented to the agent, given the prefix $s^{(k)}_{0..j}$.

$p(s^{(i)}_{j+1}|i)$ can be computed as follows: 

$p(s^{(i)}_{j+1}|i) = \frac{1}{f_j}$ where $f_j$ is the number of sequences up until the $j$th bit that are identical to the sequence. 
It is $\frac{1}{f_j}$ because the agent is still uncertain about the identity of the sequence at position $j+1$, and this uncertainty is quantified by $f_j$. 
If $f_j$ is 1 (i.e. there is only 1 item in the list up until bit j) then the agent is clearly certain about the sequence at position $j+1$, and all positions after that.
In this case, $p(s^{(i)}_{j+1}|i) = 1$.

$P(i|s^{(k)}_{0..j})$ can be computed as follows:

$P(i|s^{(k)}_{0..j}) = \frac{P(s^{(i)}_{0..j}|i)P(i)}{\sum_{i=1}^{K} P(s^{(i)}_{0..j}|i)P(i)}$
where we used Bayes' theorem to compute the posterior probability of sequence $i$ being presented, given the prefix $s^{(k)}_{0..j}$.

Now $P(s^{(i)}_{0..j}|i)=\frac{1}{f_{j-1}}}$ because the probability of the prefix $s^{(i)}_{0..j}$ being presented, given that the sequence is $i$, is
inversely proportional to the number of sequences up until the $j-1$th bit that are identical to the sequence.

Let's continue the calculation of the posterior $P(i|s^{(k)}_{0..j})$.
We have $P(i|s^{(k)}_{0..j}) = \frac{P(s^{(k)}_{0..j}|i)P(i)}{\sum_{l=1}^{K} P(s^{(k)}_{0..j}|l)P(l)}$.

Here $P(s^{(k)}_{0..j}|l)$ is the likelihood of observing prefix $s^{(k)}_{0..j}$ given that the true sequence presented has index $l$.
Based on the statement for $P(s^{(i)}_{0..j}|i)$, we can write the likelihood as:
$P(s^{(k)}_{0..j}|l) = \begin{cases} \frac{1}{f_{j-1}(s^{(l)})} & \text{if } s^{(l)}_{0..j} = s^{(k)}_{0..j} \\ 0 & \text{otherwise} \end{cases}$
where $f_{j-1}(s^{(l)})$ is the number of sequences matching the prefix $s^{(l)}_{0..j-1}$.

Let $S_k = \{ l \mid s^{(l)}_{0..j} = s^{(k)}_{0..j} \}$ be the set of indices of sequences matching the observed prefix.
The expression for the posterior becomes non-zero only for $i \in S_k$.
For $i \in S_k$:
$P(i|s^{(k)}_{0..j}) = \frac{P(s^{(k)}_{0..j}|i)P(i)}{\sum_{l \in S_k} P(s^{(k)}_{0..j}|l)P(l)}$

Substitute $P(l) = \frac{1}{Zl}$ and the likelihood expression:
$P(i|s^{(k)}_{0..j}) = \frac{\frac{1}{f_{j-1}(s^{(i)})} \frac{1}{Zi}}{\sum_{l \in S_k} \frac{1}{f_{j-1}(s^{(l)})} \frac{1}{Zl}}$

Note that for all $l \in S_k$, the prefix $s^{(l)}_{0..j-1}$ is identical to $s^{(k)}_{0..j-1}$.
Therefore, $f_{j-1}(s^{(l)})$ is the same constant value for all $l \in S_k$. Let this value be $f_{j-1}^{(k)}$.
$P(i|s^{(k)}_{0..j}) = \frac{\frac{1}{f_{j-1}^{(k)}} \frac{1}{Zi}}{\sum_{l \in S_k} \frac{1}{f_{j-1}^{(k)}} \frac{1}{Zl}} = \frac{\frac{1}{f_{j-1}^{(k)} Zi}}{\frac{1}{f_{j-1}^{(k)} Z} \sum_{l \in S_k} \frac{1}{l}}$
$P(i|s^{(k)}_{0..j}) = \frac{\frac{1}{i}}{\sum_{l \in S_k} \frac{1}{l}}$

So, the final expression for the posterior probability is:
$P(i|s^{(k)}_{0..j}) = \begin{cases} \frac{1/i}{\sum_{l \in S(s^{(k)}_{0..j})} 1/l} & \text{if } i \in S(s^{(k)}_{0..j}) \\ 0 & \text{otherwise} \end{cases}$
where $S(s^{(k)}_{0..j})$ is the set of indices $l$ such that the sequence $s^{(l)}$ starts with the observed prefix $s^{(k)}_{0..j}$.
This means the posterior probability is the prior probability $P(i)$ renormalized over the set of sequences compatible with the observed prefix.