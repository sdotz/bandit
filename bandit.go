package bandit

type Bandit interface {
	Pull() int
	Reward(int, float64)
	Reset()
}
