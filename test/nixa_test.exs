defmodule NixaTest do
  use ExUnit.Case
  doctest Nixa

  test "greets the world" do
    assert Nixa.hello() == :world
  end
end
