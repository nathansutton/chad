<?php

require_once "lib.php";

function process($x)
{
    return helper($x);
}

function run()
{
    $e = new Engine();
    return $e->process(process(1));
}
