defmodule NixaTest.Tree do
  use ExUnit.Case
  doctest Nixa

  test "ID3Classifier" do
    # Test using well-documented toy dataset for decision trees, "PlayTennis"
    {x, y} = get_play_tennis_data()
    model = Nixa.Tree.ID3Classifier.fit(x, y)
    yhat = Nixa.Tree.ID3Classifier.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    assert num_correct == Enum.count(y)
  end

  test "CARTClassifier" do
    # Test using well-documented toy dataset for decision trees, "PlayTennis"
    {x, y} = get_play_tennis_data()
    model = Nixa.Tree.CARTClassifier.fit(x, y)
    yhat = Nixa.Tree.CARTClassifier.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    assert num_correct == Enum.count(y)
  end

  test "ID3Regressor" do
    inputs = Nx.iota({10, 2}) |> Nx.to_batched_list(1)
    targets = Nx.iota({10, 1}) |> Nx.to_batched_list(1)
    model = Nixa.Tree.ID3Regressor.fit(inputs, targets)
    yhat = Nixa.Tree.ID3Regressor.predict(model, inputs) |> Nx.concatenate() |> Nx.to_flat_list()
    assert yhat == [0.5, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  end

  test "CARTRegressor" do
    inputs = Nx.iota({10, 2}) |> Nx.to_batched_list(1)
    targets = Nx.iota({10, 1}) |> Nx.to_batched_list(1)
    model = Nixa.Tree.CARTRegressor.fit(inputs, targets)
    yhat = Nixa.Tree.CARTRegressor.predict(model, inputs) |> Nx.concatenate() |> Nx.to_flat_list()
    assert yhat == [0.5, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  end

  defp get_play_tennis_data() do
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
