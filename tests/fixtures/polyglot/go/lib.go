package fixture

type Engine struct{}

func (e Engine) Process(x int) int {
	return x * 2
}

func Helper(x int) int {
	return x + 1
}
