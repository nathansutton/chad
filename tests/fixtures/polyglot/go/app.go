package fixture

func Process(x int) int {
	return Helper(x)
}

func Run() int {
	e := Engine{}
	return e.Process(Process(1))
}
