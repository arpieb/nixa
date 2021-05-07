defmodule Nixa.FeatureExtraction.Text do
  @moduledoc """
  Implements feature extraction algorithms for textual data
  """

  @doc """
  Extract a working vocabulary from a set of input sentences
  """
  def extract_vocabulary(inputs, opts \\ []) do
    tokenizer = Keyword.get(opts, :tokenizer, &String.split/1)
    stopwords = Keyword.get(opts, :stopwords, [])
    downcase = Keyword.get(opts, :downcase, true)

    inputs = if downcase,
      do: inputs |> Enum.map(&String.downcase/1),
      else: inputs

    inputs
    |> Enum.flat_map(tokenizer)
    |> MapSet.new()
    |> MapSet.difference(MapSet.new(stopwords))
  end

  @doc """
  Vectorize list of strings into count tensors using vocabulary
  """
  def count_vectorize_list(inputs, vocab) when is_list(inputs) do
    inputs
    |> Enum.map(fn s -> count_vectorize_string(s, vocab) end)
    |> Nx.concatenate()
  end

  @doc """
  Vectorize a string into a count tensor using vocabulary
  """
  def count_vectorize_string(s, vocab) when is_binary(s) do
    counts = s
      |> String.downcase()
      |> String.split()
      |> Enum.reduce(%{}, fn w, acc -> Map.update(acc, w, 1, fn x -> x + 1 end) end)
    vocab
    |> Enum.map(fn w -> Map.get(counts, w, 0) end)
    |> Nx.tensor()
    |> Nx.new_axis(0)
  end

  @doc """
  Binarize list of strings using vocabulary
  """
  def binarize_list(inputs, vocab) when is_list(inputs) do
    inputs
    |> Enum.map(fn s -> binarize_string(s, vocab) end)
    |> Nx.concatenate()
  end

  @doc """
  Binarize a string using vocabulary
  """
  def binarize_string(s, vocab) when is_binary(s) do
    counts = s
      |> String.downcase()
      |> String.split()
      |> Enum.reduce(%{}, fn w, acc -> Map.put(acc, w, 1) end)
    vocab
    |> Enum.map(fn w -> Map.get(counts, w, 0) end)
    |> Nx.tensor()
    |> Nx.new_axis(0)
  end

end
