"""NEAT config, saved genomes and the networks built from them."""

import pickle

import neat

from . import settings
from .env import NUM_INPUTS, NUM_OUTPUTS


def load_config(path=settings.NEAT_CONFIG):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))
    genome = config.genome_config
    if (genome.num_inputs, genome.num_outputs) != (NUM_INPUTS, NUM_OUTPUTS):
        raise SystemExit(f"{path.name} needs num_inputs = {NUM_INPUTS} and num_outputs = {NUM_OUTPUTS} "
                         f"(it has {genome.num_inputs} and {genome.num_outputs})")
    return config


def load_genome(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_genome(genome, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(genome, f)


def make_network(genome, config):
    return neat.nn.FeedForwardNetwork.create(genome, config)
