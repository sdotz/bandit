package bandit

import (
	"testing"

	"github.com/mroth/weightedrand/v2"
	"github.com/sdotz/bandit/epsilon_greedy"
	"github.com/sdotz/bandit/thompson_sampling"
	"github.com/sdotz/bandit/ucb"
)

type ArmProbability struct {
	Feature     string
	Probability float64
}

var foods = map[int]rune{
	0: '🍒',
	1: '🍋',
	2: '🍊',
	3: '🍉',
	4: '🥑',
}

func TestUCB1(t *testing.T) {
	chooser, _ := weightedrand.NewChooser(
		weightedrand.NewChoice(0, 0),
		weightedrand.NewChoice(1, 1),
		weightedrand.NewChoice(2, 1),
		weightedrand.NewChoice(3, 3),
		weightedrand.NewChoice(4, 5),
	)

	bandit := ucb.NewUCB1(5)

	for i := 0; i < 1_000_000; i++ {
		choice := chooser.Pick()
		pull := bandit.Pull()

		if choice == pull {
			bandit.Reward(pull, 1)
		}
	}

	for i := 0; i < len(bandit.R); i++ {
		t.Logf("%q had %f reward and %d pulls for an average reward of %f\n", foods[i], bandit.R[i], bandit.P[i], bandit.R[i]/float64(bandit.P[i]))
	}
	t.Logf("Avg Reward for UCB1: %f", bandit.TR/float64(bandit.TP))
}

func TestEpsilonGreedy(t *testing.T) {
	chooser, _ := weightedrand.NewChooser(
		weightedrand.NewChoice(0, 0),
		weightedrand.NewChoice(1, 1),
		weightedrand.NewChoice(2, 1),
		weightedrand.NewChoice(3, 3),
		weightedrand.NewChoice(4, 5),
	)

	bandit := epsilon_greedy.NewEpsilonGreedy(5, epsilon_greedy.DefaultEpsilon)

	for i := 0; i < 1_000_000; i++ {
		choice := chooser.Pick()
		pull := bandit.Pull()

		if choice == pull {
			bandit.Reward(pull, 1)
		}
	}

	for i := 0; i < len(bandit.R); i++ {
		t.Logf("%q had %f reward and %d pulls for an average reward of %f\n", foods[i], bandit.R[i], bandit.P[i], bandit.R[i]/float64(bandit.P[i]))
	}
	t.Logf("Avg Reward for Epsilon Greedy: %f", bandit.TR/float64(bandit.TP))
}

func TestThompsonSampling(t *testing.T) {
	chooser, _ := weightedrand.NewChooser(
		weightedrand.NewChoice(0, 0),
		weightedrand.NewChoice(1, 1),
		weightedrand.NewChoice(2, 1),
		weightedrand.NewChoice(3, 3),
		weightedrand.NewChoice(4, 5),
	)

	bandit := thompson_sampling.NewThompsonSampling(5)

	for i := 0; i < 1_000_000; i++ {
		choice := chooser.Pick()
		pull := bandit.Pull()

		if choice == pull {
			bandit.Reward(pull, 1)
		}
	}

	for i := 0; i < len(bandit.R); i++ {
		t.Logf("%q had %d reward and %d pulls for an average reward of %f\n", foods[i], bandit.R[i], bandit.P[i], float64(bandit.R[i])/float64(bandit.P[i]))
	}

	t.Logf("Avg Reward for Thompson Sampling: %f", float64(bandit.TR)/float64(bandit.TP))
}
