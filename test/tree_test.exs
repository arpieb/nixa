defmodule NixaTest.Tree do
  use ExUnit.Case
  doctest Nixa

  test "Classifier.ID3" do
    # Test using well-documented toy dataset for decision trees, "PlayTennis"
    {x, y} = get_play_tennis_data()
    model = Nixa.Tree.Classifier.ID3.fit(x, y)
    yhat = Nixa.Tree.Classifier.ID3.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    assert num_correct == Enum.count(y)
  end

  test "Classifier.CART" do
    # Test using well-documented toy dataset for decision trees, "PlayTennis"
    {x, y} = get_play_tennis_data()
    model = Nixa.Tree.Classifier.CART.fit(x, y)
    yhat = Nixa.Tree.Classifier.CART.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    assert num_correct == Enum.count(y)
  end

  def get_play_tennis_data() do
    {inputs, targets} = [
      { [:sunny, :hot, :high, :weak], [:no] },
      { [:sunny, :hot, :high, :strong], [:no] },
      { [:overcast, :hot, :high, :weak], [:yes] },
      { [:rain, :mild, :high, :weak], [:yes] },
      { [:rain, :cool, :normal, :weak], [:yes] },
      { [:rain, :cool, :normal, :strong], [:no] },
      { [:overcast, :cool, :normal, :strong], [:yes] },
      { [:sunny, :mild, :high, :weak], [:no] },
      { [:sunny, :cool, :normal, :weak], [:yes] },
      { [:rain, :mild, :normal, :weak], [:yes] },
      { [:sunny, :mild, :normal, :strong], [:yes] },
      { [:overcast, :mild, :high, :strong], [:yes] },
      { [:overcast, :hot, :normal, :weak], [:yes] },
      { [:rain, :mild, :high, :strong], [:no] },
      ] |> Enum.unzip()

    input_c = Nixa.Shared.get_categories(inputs)
    x = Nixa.Shared.categorical_to_numeric(inputs, input_c) |> Nx.tensor() |> Nx.to_batched_list(1)

    target_c = Nixa.Shared.get_categories(targets)
    y = Nixa.Shared.categorical_to_numeric(targets, target_c) |> Nx.tensor() |> Nx.to_batched_list(1)
    {x, y}
  end
end
