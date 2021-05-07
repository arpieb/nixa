defmodule NixaTest.NaiveBayes do
  use ExUnit.Case
  doctest Nixa

  import Nixa.FeatureExtraction.Text

  test "Categorical" do
    # Test using well-documented toy dataset for decision trees, "PlayTennis"
    {x, y} = get_play_tennis_data()
    model = Nixa.NaiveBayes.Categorical.fit(x, y)
    yhat = Nixa.NaiveBayes.Categorical.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    acc = num_correct / Enum.count(y)
    assert acc >= 0.8
  end

  test "Multinomial" do
    # Test using contrived dataset
    {x, y} = get_bow_data()
    model = Nixa.NaiveBayes.Multinomial.fit(x, y)
    yhat = Nixa.NaiveBayes.Multinomial.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    acc = num_correct / Enum.count(y)
    assert acc >= 0.8
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

  defp get_bow_data() do
    {inputs, targets} = [
      { "That food was good", [:pos] },
      { "That drink was great", [:pos] },
      { "That movie was awesome", [:pos] },
      { "That drink was bad", [:neg] },
      { "That food was awful", [:neg] },
      { "That store sucked", [:neg] },
    ] |> Enum.unzip()

    vocab = extract_vocabulary(inputs)
    x = count_vectorize_list(inputs, vocab)
      |> Nx.to_batched_list(1)

    target_c = Nixa.Shared.get_categories(targets)
    y = Nixa.Shared.categorical_to_numeric(targets, target_c)
      |> Nx.tensor()
      |> Nx.to_batched_list(1)

    {x, y}
  end

end
