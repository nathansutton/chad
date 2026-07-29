fun process(x: Int): Int {
    return helper(x)
}

fun run(): Int {
    return Engine().process(process(1))
}
