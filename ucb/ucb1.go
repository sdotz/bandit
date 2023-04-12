package ucb

import "math"

// UCB1 implements the UCB1 algorithm for multi-armed bandits.
// See http://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf for details.
type UCB1 struct {
	// Number of times each arm has been pulled.
	P []int
	// Cumulative reward for each arm.
	R []float64
	// Total number of pulls.
	TP int
	TR float64
}

// NewUCB1 returns a new UCB1 instance with K arms.
func NewUCB1(K int) *UCB1 {
	return &UCB1{
		P: make([]int, K),
		R: make([]float64, K),
	}
}

// Pull pulls arm i and returns the index
func (u *UCB1) Pull() int {
	pulledArm := u.bestArm()
	u.TP++
	u.P[pulledArm]++
	return pulledArm

}

// BestArm returns the index of the best arm.
func (u *UCB1) bestArm() int {
	//Try all arms at first by pulling the next not pulled arm
	if u.TP < len(u.P) {
		return u.TP
	}

	best := 0.0
	bestIdx := 0
	for i := 1; i < len(u.P); i++ {
		score := u.R[i]/float64(u.P[i]) + math.Sqrt(2*math.Log(float64(u.TP))/float64(u.P[i]))
		if score > best {
			best = score
			bestIdx = i
		}
	}
	return bestIdx
}

// Reward updates the algorithm with the reward for arm i.
func (u *UCB1) Reward(armIdx int, reward float64) {
	u.R[armIdx] += reward
	u.TR++
}

// Reset resets the algorithm.
func (u *UCB1) Reset() {
	u.P = make([]int, len(u.P))
	u.R = make([]float64, len(u.P))
	u.TP = 0
	u.TR = 0
}
