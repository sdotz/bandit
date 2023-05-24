package thompson_sampling

import (
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type ThompsonSampling struct {
	// Number of rewards for each arm.
	R []int
	// Number of pulls for each arm.
	P []int
	// Total number of pulls.
	TP int
	// Total rewards.
	TR int
}

// NewThompsonSampling returns a new ThompsonSampling instance with K arms.
func NewThompsonSampling(K int) *ThompsonSampling {
	return &ThompsonSampling{
		R: make([]int, K),
		P: make([]int, K),
	}
}

// Pull selects an arm using Thompson Sampling and returns the index.
func (t *ThompsonSampling) Pull() int {
	sampledValues := make([]float64, len(t.R))
	for i := 0; i < len(t.R); i++ {
		betaDist := distuv.Beta{
			Alpha: float64(t.R[i]) + 1,
			Beta:  float64(t.P[i]) + 1,
			Src:   rand.New(rand.NewSource(uint64(time.Now().UnixNano()))),
		}
		sampledValues[i] = betaDist.Rand()
	}
	t.TP++
	pulledIdx := maxIndex(sampledValues)
	t.P[pulledIdx]++
	return pulledIdx
}

// Reward updates the Thompson Sampling algorithm with the reward for arm i.
func (t *ThompsonSampling) Reward(armIdx int, reward int) {
	if reward == 1 {
		t.R[armIdx]++
		t.TR++
	} else {
		panic("Reward must be 0 or 1.")
	}
}

// Reset resets the ThompsonSampling instance.
func (t *ThompsonSampling) Reset() {
	t.P = make([]int, len(t.P))
	t.R = make([]int, len(t.R))
	t.TP = 0
	t.TR = 0
}

// maxIndex returns the index of the maximum value in a slice.
func maxIndex(slice []float64) int {
	max := slice[0]
	maxIndex := 0
	for i := 1; i < len(slice); i++ {
		if slice[i] > max {
			max = slice[i]
			maxIndex = i
		}
	}
	return maxIndex
}
