"""Tests for eval.load_embeddings."""

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.eval.load_embeddings import EmbedderSpec, load_embedder_matrix

VIDEO_A = 'Day1_Tank1_Cuttle1_Resident_Crop'
VIDEO_B = 'Day1_Tank1_Cuttle2_Intruder_Crop'


class TestLoadEmbedderMatrix:
    """Test the function load_embedder_matrix."""

    def test_restricts_and_orders_to_manifest(
        self, tmp_path, make_latents_dir, write_model_config,
    ):
        # Arrange
        model_dir = tmp_path / 'beast_models' / 'my-model'
        write_model_config(model_dir, model_class='resnet')
        make_latents_dir(model_dir, {
            VIDEO_A: {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])},
            VIDEO_B: {0: np.array([5.0, 6.0])},
        })
        manifest = pd.DataFrame({
            'video_name': [VIDEO_B, VIDEO_A],
            'frame_number': [0, 1],
        })
        spec = EmbedderSpec(id='my-model', model_dir=model_dir)

        # Act
        X = load_embedder_matrix(spec, manifest)

        # Assert
        np.testing.assert_array_equal(X, [[5.0, 6.0], [3.0, 4.0]])

    def test_msps_vae_subspace_selection(self, tmp_path, make_latents_dir, write_model_config):
        # Arrange
        model_dir = tmp_path / 'beast_models' / 'my-msps-vae'
        write_model_config(
            model_dir, model_class='msps_vae',
            num_latents_unsupervised=2, num_latents_background=1,
        )
        make_latents_dir(model_dir, {VIDEO_A: {0: np.array([1.0, 2.0, 3.0])}})
        manifest = pd.DataFrame({'video_name': [VIDEO_A], 'frame_number': [0]})

        # Act
        z_u = load_embedder_matrix(
            EmbedderSpec(id='z_u', model_dir=model_dir, subspace='unsupervised'), manifest,
        )
        z_b = load_embedder_matrix(
            EmbedderSpec(id='z_b', model_dir=model_dir, subspace='background'), manifest,
        )

        # Assert
        np.testing.assert_array_equal(z_u, [[1.0, 2.0]])
        np.testing.assert_array_equal(z_b, [[3.0]])

    def test_all_subspace_returns_full_vector_for_msps_vae(
        self, tmp_path, make_latents_dir, write_model_config,
    ):
        # Arrange: split_latent_spaces has no 'all' key for msps_vae -- 'all' must be
        # handled directly rather than delegated to it.
        model_dir = tmp_path / 'beast_models' / 'my-msps-vae'
        write_model_config(
            model_dir, model_class='msps_vae',
            num_latents_unsupervised=2, num_latents_background=1,
        )
        make_latents_dir(model_dir, {VIDEO_A: {0: np.array([1.0, 2.0, 3.0])}})
        manifest = pd.DataFrame({'video_name': [VIDEO_A], 'frame_number': [0]})

        # Act
        z_all = load_embedder_matrix(EmbedderSpec(id='z_all', model_dir=model_dir), manifest)

        # Assert
        np.testing.assert_array_equal(z_all, [[1.0, 2.0, 3.0]])

    def test_raises_on_missing_frame(self, tmp_path, make_latents_dir, write_model_config):
        # Arrange
        model_dir = tmp_path / 'beast_models' / 'my-model'
        write_model_config(model_dir, model_class='resnet')
        make_latents_dir(model_dir, {VIDEO_A: {0: np.array([1.0])}})
        manifest = pd.DataFrame({'video_name': [VIDEO_A], 'frame_number': [99]})
        spec = EmbedderSpec(id='my-model', model_dir=model_dir)

        # Act / Assert
        with pytest.raises(ValueError, match='no matching latent vector'):
            load_embedder_matrix(spec, manifest)
