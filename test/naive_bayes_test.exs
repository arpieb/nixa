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
    assert acc >= 0.75
  end

  test "Multinomial" do
    # Test using contrived dataset
    {x, y} = get_vectorized_data()
    model = Nixa.NaiveBayes.Multinomial.fit(x, y)

    yhat = Nixa.NaiveBayes.Multinomial.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    acc = num_correct / Enum.count(y)
    assert acc >= 0.8
  end

  test "Bernoulli" do
    # Test using contrived dataset
    {x, y} = get_binarized_data()
    model = Nixa.NaiveBayes.Bernoulli.fit(x, y)

    yhat = Nixa.NaiveBayes.Bernoulli.predict(model, x) |> Nx.concatenate() |> Nx.squeeze()
    ytrue = y |> Nx.concatenate() |> Nx.squeeze()

    num_correct = Nx.equal(yhat, ytrue) |> Nx.sum() |> Nx.to_scalar()
    acc = num_correct / Enum.count(y)
    assert acc >= 0.8
  end

  test "Gaussian" do
    # Test using data from:
    # https://towardsdatascience.com/learning-by-implementing-gaussian-naive-bayes-3f0e3d2c01b2
    {x, y} = get_gaussian_data()
    model = Nixa.NaiveBayes.Gaussian.fit(x, y)

    x_test = Nx.tensor([[-2, 5]]) |> Nx.to_batched_list(1)
    yhat = Nixa.NaiveBayes.Gaussian.predict(model, x_test) |> Nx.concatenate() |> Nx.squeeze() |> Nx.to_scalar()
    assert yhat == 3
  end

  ### Internal functions

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

  defp get_vectorized_data() do
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

  defp get_binarized_data() do
    {inputs, targets} = [
      { "That food was good", [:pos] },
      { "That drink was great", [:pos] },
      { "That movie was awesome", [:pos] },
      { "That drink was bad", [:neg] },
      { "That food was awful", [:neg] },
      { "That store sucked", [:neg] },
    ] |> Enum.unzip()

    vocab = extract_vocabulary(inputs)
    x = binarize_list(inputs, vocab) |> Nx.to_batched_list(1)

    target_c = Nixa.Shared.get_categories(targets)
    y = Nixa.Shared.categorical_to_numeric(targets, target_c) |> Nx.tensor() |> Nx.to_batched_list(1)

    {x, y}
  end

  defp get_gaussian_data() do
    x = Nx.tensor([
      [-5.34791215,  5.15634897],
      [ 0.14404357,  1.45427351],
      [ 0.97873798,  2.2408932 ],
      [ 1.86755799, -0.97727788],
      [ 5.3130677,   4.14590426],
      [ 5.44386323,  5.33367433],
      [ 1.76405235,  0.40015721],
      [ 6.49407907,  4.79484174],
      [-5.88778575,  3.01920353],
      [-3.76970932,  6.20237985],
      [-5.38732682,  4.69769725],
      [ 5.04575852,  4.81281615],
      [ 0.76103773,  0.12167502],
      [-4.84505257,  5.37816252],
      [ 5.8644362,   4.25783498],
      [-0.10321885,  0.4105985 ],
      [ 2.44701018,  5.6536186 ],
      [ 7.26975462,  3.54563433],
      [ 0.95008842, -0.15135721],
      [-3.46722079,  6.46935877]
      ]) |> Nx.to_batched_list(1)
    y = Nx.tensor([[2], [0], [0], [0], [1], [1], [0], [1], [2], [2], [2], [1], [0], [2], [1], [0], [1], [1], [0], [2]]) |> Nx.to_batched_list(1)
    {x, y}
  end

end
