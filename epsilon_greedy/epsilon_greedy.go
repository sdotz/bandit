package epsilon_greedy

import "math/rand"

// EpsilonGreedy implements the epsilon-greedy algorithm for multi-armed bandits.

// See https://en.wikipedia.org/wiki/Multi-armed_bandit#Epsilon-greedy for details.

const (
	// Epsilon is the probability of choosing a random arm.
	DefaultEpsilon = 0.1
)

type EpsilonGreedy struct {
	// Number of times each arm has been pulled.
	P []int
	// Cumulative reward for each arm.
	R []float64
	// Total number of pulls.
	TP      int
	TR      float64
	Epsilon float64
}

// NewEpsilonGreedy returns a new EpsilonGreedy instance with K arms.

func NewEpsilonGreedy(K int, epsilon float64) *EpsilonGreedy {
	return &EpsilonGreedy{
		P:       make([]int, K),
		R:       make([]float64, K),
		Epsilon: epsilon,
	}
}

// Pull pulls arm i and returns the index

func (e *EpsilonGreedy) Pull() int {
	pulledArm := e.bestArm()
	e.TP++
	e.P[pulledArm]++
	return pulledArm

}

// BestArm returns the index of the best arm.

func (e *EpsilonGreedy) bestArm() int {
	//explore with probability epsilon
	if rand.Float64() < e.Epsilon {
		return rand.Intn(len(e.P))
	}

	//exploit with probability 1-epsilon the arm with the best avg reward
	best := 0.0
	bestIdx := 0
	for i := 1; i < len(e.P); i++ {
		score := e.R[i] / float64(e.P[i])
		if score > best {
			best = score
			bestIdx = i
		}
	}
	return bestIdx
}

// Reward updates the algorithm with the reward for arm i.

func (e *EpsilonGreedy) Reward(armIdx int, reward float64) {
	e.R[armIdx] += reward
	e.TR++
}

// Reset resets the algorithm.
func (e *EpsilonGreedy) Reset() {
	e.P = make([]int, len(e.P))
	e.R = make([]float64, len(e.R))
	e.TP = 0
	e.TR = 0
}
