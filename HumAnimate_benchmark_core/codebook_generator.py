from pathlib import Path
from utils import get_config

import pandas as pd


class TextPoseBench_CodeBook_Generator(object):
    def __init__(self, benchmark_config, pose_config, prompt_config):
        self.bench_cfg = benchmark_config
        self.pose_config = pose_config
        self.prompt_config = prompt_config
        self.save_path = Path(self.bench_cfg["save_path"])
        self.codebook_name = self.bench_cfg["codebook_name"]
        self.pose_seq_name_placeholder = "{}_{}_{}_{}_{}_{}.npy" # <filename>_<pose_type>_<fps>_<height>_<translation>_<misc>
        self.prompt_placeholder = "{} {} {}." # <person> <action_description> <place>
        self.codebook = self.instantiate_codebook()

    def instantiate_codebook(self):
        data = {
            "filename": list(),
            "prompt": list(),
            "original_vid_fps": list(),
            "sampling_fps": list(),
            "pose_height_class": list(),
            "pose_height_value": list(),
            "translation": list(),
            "pose_misc": list(),
            "pose_metadata_filename": list(),
            "person_placeholder": list(),
            "action_placeholder": list(),
            "place_placeholder": list(),
            "experiment_comment": list()
        }
        return data

    def add_to_table(self,
                     filename,
                     prompt,
                     original_vid_fps,
                     sampling_fps,
                     pose_height_class,
                     pose_height_value,
                     translation,
                     pose_misc,
                     pose_metadata_filename,
                     person_placeholder,
                     action_placeholder,
                     place_placeholder,
                     experiment_comment):
        self.codebook["filename"].append(filename)
        self.codebook["prompt"].append(prompt)
        self.codebook["original_vid_fps"].append(original_vid_fps)
        self.codebook["sampling_fps"].append(sampling_fps)
        self.codebook["pose_height_class"].append(pose_height_class)
        self.codebook["pose_height_value"].append(pose_height_value)
        self.codebook["translation"].append(translation)
        self.codebook["pose_misc"].append(pose_misc)
        self.codebook["pose_metadata_filename"].append(pose_metadata_filename)
        self.codebook["person_placeholder"].append(person_placeholder)
        self.codebook["action_placeholder"].append(action_placeholder)
        self.codebook["place_placeholder"].append(place_placeholder)
        self.codebook["experiment_comment"].append(experiment_comment)

    def collect_all_action_default_set(self, pose_style, experiment_comment=None):
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"] + self.bench_cfg["pose_sequences"]["advanced_set"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        person_default_verbosity = self.bench_cfg["defaults"]["person_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            pose_default_height,
                                            pose_default_translation,
                                            pose_default_misc
                                        )
            action = self.pose_config[pose_seq]["action"]
            for person_ref in prompt_default_persons:
                person_default = self.prompt_config["verbosity"]["fg"][person_ref][person_default_verbosity]
                for background_ref in prompt_default_backgrounds:
                    background_default = self.prompt_config["verbosity"]["bg"][background_ref][bg_default_verbosity]
                    prompt = self.prompt_placeholder.format(
                                            person_default,
                                            action,
                                            background_default
                                            )
                    self.add_to_table(
                        filename=filename,
                        prompt=prompt,
                        original_vid_fps=pose_config[pose_seq]["fps"],
                        sampling_fps=pose_default_fps,
                        pose_height_class=pose_default_height,
                        pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                        translation=pose_default_translation,
                        pose_misc=pose_default_misc,
                        pose_metadata_filename=pose_seq,
                        person_placeholder=person_default,
                        action_placeholder=action,
                        place_placeholder=background_default,
                        experiment_comment=experiment_comment,
                    )

    def collect_camera_motion_settings(self, pose_style, experiment_comment=None, default_collected=True):
        pose_translations = self.bench_cfg["pose_translation_options"]["modes"]
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        person_default_verbosity = self.bench_cfg["defaults"]["person_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            for translation in pose_translations:
                if default_collected and (translation == pose_default_translation):
                    continue
                filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            pose_default_height,
                                            translation,
                                            pose_default_misc
                    )
                action = self.pose_config[pose_seq]["action"]
                for person_ref in prompt_default_persons:
                    person_default = self.prompt_config["verbosity"]["fg"][person_ref][person_default_verbosity]
                    additional_prompt = self.bench_cfg["pose_translation_options"]["misc_prompt"][translation]
                    for bg_ref in prompt_default_backgrounds:
                        bg_default = self.prompt_config["verbosity"]["bg"][bg_ref][bg_default_verbosity]
                        prompt = self.prompt_placeholder.format(
                                            person_default,
                                            action,
                                            bg_default
                            )
                        prompt += additional_prompt
                    self.add_to_table(
                        filename=filename,
                        prompt=prompt,
                        original_vid_fps=pose_config[pose_seq]["fps"],
                        sampling_fps=pose_default_fps,
                        pose_height_class=pose_default_height,
                        pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                        translation=translation,
                        pose_misc=pose_default_misc,
                        pose_metadata_filename=pose_seq,
                        person_placeholder=person_default,
                        action_placeholder=action,
                        place_placeholder=bg_default,
                        experiment_comment=experiment_comment,
                    )

    def collect_fps_experiment_settings(self, pose_style, experiment_comment=None, default_collected=True):
        pose_sampling_fps = self.bench_cfg["pose_fps_options"]
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"] + self.bench_cfg["pose_sequences"]["advanced_set"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        default_fps_to_skip = self.bench_cfg["defaults"]["fps"]
        person_default_verbosity = self.bench_cfg["defaults"]["person_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            for fps in pose_sampling_fps:
                if default_collected and (fps == default_fps_to_skip):
                    continue
                filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            fps,
                                            pose_default_height,
                                            pose_default_translation,
                                            pose_default_misc
                    )
                action = self.pose_config[pose_seq]["action"]
                for person_ref in prompt_default_persons:
                    person_default = self.prompt_config["verbosity"]["fg"][person_ref][person_default_verbosity]
                    for bg_ref in prompt_default_backgrounds:
                        bg_default = self.prompt_config["verbosity"]["bg"][bg_ref][bg_default_verbosity]
                        prompt = self.prompt_placeholder.format(
                                            person_default,
                                            action,
                                            bg_default
                            )
                    self.add_to_table(
                        filename=filename,
                        prompt=prompt,
                        original_vid_fps=pose_config[pose_seq]["fps"],
                        sampling_fps=fps,
                        pose_height_class=pose_default_height,
                        pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                        translation=pose_default_translation,
                        pose_misc=pose_default_misc,
                        pose_metadata_filename=pose_seq,
                        person_placeholder=person_default,
                        action_placeholder=action,
                        place_placeholder=bg_default,
                        experiment_comment=experiment_comment,
                    )

    def collect_pose_size_experiment_settings(self, pose_style, experiment_comment=None, default_collected=True):
        pose_heights = self.bench_cfg["pose_height_options"]
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"] + self.bench_cfg["pose_sequences"]["advanced_set"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        default_height_to_skip = self.bench_cfg["defaults"]["height"]
        person_default_verbosity = self.bench_cfg["defaults"]["person_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            for height in pose_heights:
                if default_collected and (height == default_height_to_skip):
                    continue
                filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            height,
                                            pose_default_translation,
                                            pose_default_misc
                    )
                action = self.pose_config[pose_seq]["action"]
                for person_ref in prompt_default_persons:
                    person_default = self.prompt_config["verbosity"]["fg"][person_ref][person_default_verbosity]
                    for bg_ref in prompt_default_backgrounds:
                        bg_default = self.prompt_config["verbosity"]["bg"][bg_ref][bg_default_verbosity]
                        prompt = self.prompt_placeholder.format(
                                            person_default,
                                            action,
                                            bg_default
                            )
                    self.add_to_table(
                        filename=filename,
                        prompt=prompt,
                        original_vid_fps=pose_config[pose_seq]["fps"],
                        sampling_fps=pose_default_fps,
                        pose_height_class=height,
                        pose_height_value=pose_config[pose_seq]["height_values"][height],
                        translation=pose_default_translation,
                        pose_misc=pose_default_misc,
                        pose_metadata_filename=pose_seq,
                        person_placeholder=person_default,
                        action_placeholder=action,
                        place_placeholder=bg_default,
                        experiment_comment=experiment_comment,
                    )

    def collect_manually_altered_pose_experiment_settings(self, pose_style, experiment_comment=None, default_collected=True):
        options = self.bench_cfg["pose_namually_edited_options"]
        pose_sequences = options["pose_seq"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        person_default_verbosity = self.bench_cfg["defaults"]["person_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            pose_misc = options[pose_seq]
            for misc in pose_misc:
                if default_collected and (misc == pose_default_misc):
                    continue
                filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            pose_default_height,
                                            pose_default_translation,
                                            misc
                    )
                action = self.pose_config[pose_seq]["action"]
                if pose_seq in self.bench_cfg["pose_namually_edited_options"]["special_prompt"].keys():
                    action = self.bench_cfg["pose_namually_edited_options"]["special_prompt"][pose_seq]
                for person_ref in prompt_default_persons:
                    person_default = self.prompt_config["verbosity"]["fg"][person_ref][person_default_verbosity]
                    for bg_ref in prompt_default_backgrounds:
                        bg_default = self.prompt_config["verbosity"]["bg"][bg_ref][bg_default_verbosity]
                        prompt = self.prompt_placeholder.format(
                                            person_default,
                                            action,
                                            bg_default
                            )
                    self.add_to_table(
                        filename=filename,
                        prompt=prompt,
                        original_vid_fps=pose_config[pose_seq]["fps"],
                        sampling_fps=pose_default_fps,
                        pose_height_class=pose_default_height,
                        pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                        translation=pose_default_translation,
                        pose_misc=misc,
                        pose_metadata_filename=pose_seq,
                        person_placeholder=person_default,
                        action_placeholder=action,
                        place_placeholder=bg_default,
                        experiment_comment=experiment_comment,
                    )

    def collect_prompt_verbosity_settings(self, pose_style, experiment_comment=None, default_collected=False):
        verbosity_levels = self.bench_cfg["prompt_verbosity_options"]
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"] + self.bench_cfg["pose_sequences"]["advanced_set"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        person_default_verbosity = self.bench_cfg["defaults"]["person_verbosity"]
        background_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            pose_default_height,
                                            pose_default_translation,
                                            pose_default_misc
                                        )
            action = self.pose_config[pose_seq]["action"]
            for person_ref in prompt_default_persons:
                for verbosity in verbosity_levels:
                    if default_collected and (verbosity == person_default_verbosity):
                        continue
                    person = self.prompt_config["verbosity"]["fg"][person_ref][verbosity]
                    for background_ref in prompt_default_backgrounds:
                        background_default = self.prompt_config["verbosity"]["bg"][background_ref][bg_default_verbosity]
                        prompt = self.prompt_placeholder.format(
                                                person,
                                                action,
                                                background_default
                                                )
                        self.add_to_table(
                            filename=filename,
                            prompt=prompt,
                            original_vid_fps=pose_config[pose_seq]["fps"],
                            sampling_fps=pose_default_fps,
                            pose_height_class=pose_default_height,
                            pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                            translation=pose_default_translation,
                            pose_misc=pose_default_misc,
                            pose_metadata_filename=pose_seq,
                            person_placeholder=person,
                            action_placeholder=action,
                            place_placeholder=background_default,
                            experiment_comment=experiment_comment,
                        )
            for person_ref in prompt_default_persons:
                person_default = self.prompt_config["verbosity"]["fg"][person_ref][person_default_verbosity]
                for verbosity in verbosity_levels:
                    if default_collected and (verbosity == background_default_verbosity):
                        continue
                    for background_ref in prompt_default_backgrounds:
                        background = self.prompt_config["verbosity"]["bg"][background_ref][verbosity]
                        prompt = self.prompt_placeholder.format(
                                                person_default,
                                                action,
                                                background
                                                )
                        self.add_to_table(
                            filename=filename,
                            prompt=prompt,
                            original_vid_fps=pose_config[pose_seq]["fps"],
                            sampling_fps=pose_default_fps,
                            pose_height_class=pose_default_height,
                            pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                            translation=pose_default_translation,
                            pose_misc=pose_default_misc,
                            pose_metadata_filename=pose_seq,
                            person_placeholder=person_default,
                            action_placeholder=action,
                            place_placeholder=background,
                            experiment_comment=experiment_comment,
                        )

    def collect_person_specificness_settings(self, pose_style, experiment_comment=None, default_collected=True):
        person_specificness_levels = self.bench_cfg["prompt_person_specificness_options"]
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"] + self.bench_cfg["pose_sequences"]["advanced_set"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        person_default_spec = self.bench_cfg["defaults"]["person_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            pose_default_height,
                                            pose_default_translation,
                                            pose_default_misc
                                        )
            action = self.pose_config[pose_seq]["action"]
            for person_ref in prompt_default_persons:
                for specificness in person_specificness_levels:
                    if default_collected and (specificness == person_default_spec):
                        continue
                    for person in self.prompt_config["person_specificness"][person_ref][specificness]:
                        for background_ref in prompt_default_backgrounds:
                            background_default = self.prompt_config["verbosity"]["bg"][background_ref][bg_default_verbosity]
                            prompt = self.prompt_placeholder.format(
                                                    person,
                                                    action,
                                                    background_default
                                                    )
                            self.add_to_table(
                                filename=filename,
                                prompt=prompt,
                                original_vid_fps=pose_config[pose_seq]["fps"],
                                sampling_fps=pose_default_fps,
                                pose_height_class=pose_default_height,
                                pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                                translation=pose_default_translation,
                                pose_misc=pose_default_misc,
                                pose_metadata_filename=pose_seq,
                                person_placeholder=person,
                                action_placeholder=action,
                                place_placeholder=background_default,
                                experiment_comment=experiment_comment,
                            )

    def collect_person_shape_experiment_settings(self, pose_style, experiment_comment=None, default_collected=True):
        person_shape_specifications = self.bench_cfg["prompt_person_shape_control_options"]
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"] + self.bench_cfg["pose_sequences"]["advanced_set"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        person_default_spec = self.bench_cfg["defaults"]["person_verbosity"]
        bg_default_verbosity = self.bench_cfg["defaults"]["background_verbosity"]
        for pose_seq in pose_sequences:
            filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            pose_default_height,
                                            pose_default_translation,
                                            pose_default_misc
                                        )
            action = self.pose_config[pose_seq]["action"]
            for person_ref in prompt_default_persons:
                for shape_type in person_shape_specifications:
                    if default_collected and (shape_type == person_default_spec):
                        continue
                    for person in self.prompt_config["person_shape_controls"][person_ref][shape_type]:
                        for background_ref in prompt_default_backgrounds:
                            background_default = self.prompt_config["verbosity"]["bg"][background_ref][bg_default_verbosity]
                            prompt = self.prompt_placeholder.format(
                                                    person,
                                                    action,
                                                    background_default
                                                    )
                            self.add_to_table(
                                filename=filename,
                                prompt=prompt,
                                original_vid_fps=pose_config[pose_seq]["fps"],
                                sampling_fps=pose_default_fps,
                                pose_height_class=pose_default_height,
                                pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                                translation=pose_default_translation,
                                pose_misc=pose_default_misc,
                                pose_metadata_filename=pose_seq,
                                person_placeholder=person,
                                action_placeholder=action,
                                place_placeholder=background_default,
                                experiment_comment=experiment_comment,
                            )

    def collect_background_specificness_exp_settings(self, pose_style, experiment_comment=None, default_collected=True):
        background_specifications = self.bench_cfg["prompt_background_specificness_options"]
        pose_sequences = self.bench_cfg["pose_sequences"]["base_set"] + self.bench_cfg["pose_sequences"]["advanced_set"]
        pose_default_fps = self.bench_cfg["defaults"]["fps"]
        pose_default_height = self.bench_cfg["defaults"]["height"]
        pose_default_translation = self.bench_cfg["defaults"]["translation"]
        pose_default_misc = self.bench_cfg["defaults"]["misc"]
        prompt_default_persons = self.bench_cfg["defaults"]["person_references"]
        prompt_default_backgrounds = self.bench_cfg["defaults"]["background_references"]
        person_default_verbosity = self.bench_cfg["defaults"]["person_verbosity"]
        prompt_default_bg_list = [self.prompt_config["verbosity"]["bg"][ref]["default"] for ref in prompt_default_backgrounds]
        for pose_seq in pose_sequences:
            filename = self.pose_seq_name_placeholder.format(
                                            pose_seq,
                                            pose_style,
                                            pose_default_fps,
                                            pose_default_height,
                                            pose_default_translation,
                                            pose_default_misc
                                        )
            action = self.pose_config[pose_seq]["action"]
            for person_ref in prompt_default_persons:
                person_default = self.prompt_config["verbosity"]["fg"][person_ref][person_default_verbosity]
                for bg_spec in background_specifications:
                    for background in self.prompt_config["background_specificness"][bg_spec]:
                        if default_collected and (background in prompt_default_bg_list):
                            continue
                        prompt = self.prompt_placeholder.format(
                                                person_default,
                                                action,
                                                background
                                                )
                        self.add_to_table(
                            filename=filename,
                            prompt=prompt,
                            original_vid_fps=pose_config[pose_seq]["fps"],
                            sampling_fps=pose_default_fps,
                            pose_height_class=pose_default_height,
                            pose_height_value=pose_config[pose_seq]["height_values"][pose_default_height],
                            translation=pose_default_translation,
                            pose_misc=pose_default_misc,
                            pose_metadata_filename=pose_seq,
                            person_placeholder=person_default,
                            action_placeholder=action,
                            place_placeholder=background,
                            experiment_comment=experiment_comment,
                        )

    def create_codebook(self):
        pose_styles = self.bench_cfg["pose_seq_styles"]
        for pose_style in pose_styles:
            self.codebook = self.instantiate_codebook()
            # self.collect_camera_motion_settings(pose_style, experiment_comment="camera_motion", default_collected=True) # 160 (10*8*2)
            # self.collect_all_action_default_set(pose_style, experiment_comment="default") # 32 (16*2)
            # self.collect_fps_experiment_settings(pose_style, experiment_comment="fps", default_collected=True) # 64 (16*3*2 - (16*2)) # normal
            # self.collect_pose_size_experiment_settings(pose_style, experiment_comment="pose_size", default_collected=True) # 64 (16*3*2 - (16*2)) # 5fps
            # self.collect_prompt_verbosity_settings(pose_style, experiment_comment="prompt_verbosity", default_collected=True) # 64 (16*3*2 - (16*2)) "default"
            # self.collect_person_specificness_settings(pose_style, experiment_comment="person_specificness", default_collected=True) # 256 (16*18 - 16*2) default
            # self.collect_person_shape_experiment_settings(pose_style, experiment_comment="person_shape_control", default_collected=True) # 128 (16*10 - 16*2) default
            # self.collect_background_specificness_exp_settings(pose_style, experiment_comment="background_specificness", default_collected=True) # 352 (16*12*2 - 16*2) "in a park"
            self.collect_manually_altered_pose_experiment_settings(pose_style, experiment_comment="manually_altered_pose_seq", default_collected=True) # 10 (2*2*2 + 2)

            save_filename = self.codebook_name+"__{}.csv".format(pose_style)
            df = pd.DataFrame(self.codebook)
            df.to_csv(str(self.save_path/save_filename))
            print("Generated the benchmark codebook at {} with {} rows".format(self.save_path/save_filename, len(self.codebook["prompt"])))


if __name__ == "__main__":
    benchmark_config = get_config("configs/benchmark_info.yaml")
    pose_config = get_config("configs/pose_sequences_info.yaml")
    prompt_config = get_config("configs/prompt_info.yaml")

    obj = TextPoseBench_CodeBook_Generator(benchmark_config, pose_config, prompt_config)
    obj.create_codebook()
