import sys

mode = sys.argv[1]
print(mode)
# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from pl_model import Text_Mmamba_pl
# import lightning as L
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# others
from glob import glob
import numpy as np
import os
import json
from tqdm import tqdm
import math
# import argparse
from transformers import T5EncoderModel, T5Tokenizer
# from text_simba import MB_Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.multiprocessing.set_start_method('spawn')
from utils import *

def create_logger(logger_file_path, name=None):
    import time
    import logging
    
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    if name is not None:
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_name = '{}.log'.format(name)
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

model_path = sys.argv[1]
folder_name = sys.argv[2]
subfolder_name = sys.argv[3]


config_path = model_path[::-1].split('/', 4)[-1][::-1]+'/config.json'
with open(config_path) as f:
    config = json.load(f)
model = Text_Mmamba_pl.load_from_checkpoint(model_path, config)
model.eval()
model.freeze()
# folder_name = 'musicgen_baseline'
save_path = f'/mnt/gestalt/home/lonian/mamba/listening_test_results_demo/{folder_name}/{subfolder_name}/dac_token'
os.makedirs (save_path, exist_ok=True)

logger = create_logger(f'/mnt/gestalt/home/lonian/mamba/listening_test_results_demo/{folder_name}/{subfolder_name}')

logger.info(f'Is incontext: {config['model']['is_incontext']}')
logger.info(f'Attention layers: {config['model']['self_atten_layers']}')
logger.info(f'Is pure mamba: {config['model']['is_pure_mamba']}')
logger.info(model_path)

# import datasets
# datasets_200 = datasets.load_from_disk('/mnt/gestalt/home/lonian/datasets/MusicCaps/eval_sub_50_prompt')
# loader = DataLoader(dataset=datasets_200, batch_size = 5)

captions = [
            # 'A fast-paced electronic track with pounding kick drums, rolling basslines, and evolving synth sequences. The relentless groove, layered with atmospheric effects, builds tension and energy, perfect for a club setting.',
            # 'A groovy house track featuring punchy kick drums, smooth basslines, and uplifting synth chords. The steady four-on-the-floor beat keeps the momentum, while shimmering pads and subtle FX create a hypnotic dancefloor vibe.',
            # 'A high-energy rock instrumental driven by electrifying guitar riffs, powerful drum beats, and dynamic bass grooves. The track builds with intensity, featuring soaring solos and a raw, rebellious spirit that ignites adrenaline.',
            # 'A smooth jazz composition with mellow saxophone melodies, walking basslines, and intricate piano harmonies. The subtle swing rhythm and expressive instrumental solos create a laid-back yet sophisticated mood.',
            # "This song seems to be experimental. An e-bass is playing a light funky groove along with a clapping sound while other percussion's are adding a rhythm rounding up the rhythm section. A oriental sounding string is playing one single note rhythmically while another oriental and plucked instrument is playing a melody in the higher register. The whole recording sounds a little old but is of good quality. This song may be playing in the kitchen while cooking.",
            # 'This is the recording of a jazz reggae concert. There is a saxophone lead playing a solo. There is a keyboard and an electric guitar playing the main tune with the backing of a bass guitar. In the rhythmic background, there is an acoustic reggae drum beat. The atmosphere is groovy and chill. This piece could be playing in the background at a beach. It could also be included in the soundtrack of a summer/vacation/tropical themed movie.'
            # 'A bluesy acoustic guitar solo with expressive slides and fingerpicked melodies. The warm tone of the guitar is complemented by a subtle shuffle rhythm in the background, creating a nostalgic and intimate mood. This piece would suit a road trip scene or a cozy evening by the fireplace.',
            # 'A deep house track with a hypnotic bassline, and a steady kick-clap rhythm. The synth pads create a lush and immersive atmosphere, while subtle effects add movement. This could be playing in a high-end lounge or during a late-night DJ set.',
            # 'A reggae instrumental with a relaxed groove, featuring offbeat electric guitar skanks, a deep, laid-back bassline, and a gentle horn section. The rhythm section provides a smooth bounce, creating a sunny and carefree vibe. This track would be perfect for a tropical island setting or a beachside café.',
            # 'A funky jazz-fusion groove featuring a slap bass riff, syncopated electric piano chords, and lively brass stabs. The drums maintain a tight rhythm with ghost notes on the snare, keeping the energy high. This piece would suit a late-night urban scene or a stylish heist film.',
            # 'A heavy rock instrumental driven by distorted electric guitars and a powerful drum groove. The bassline adds weight to the mix, while an energetic guitar solo shreds in the forefront. This song would be fitting for an intense action sequence or a high-adrenaline sports montage.',
            # 'A melancholic solo piano piece with delicate, flowing melodies and soft pedal resonance. The tempo is slow, and the dynamics shift gently, evoking introspection and deep emotion. This could be playing in the background of a heartfelt movie scene or during a quiet, rainy evening.',
            # 'A high-energy electronic dance track with a pulsating four-on-the-floor kick drum and shimmering hi-hats. Deep synth bass drives the rhythm, while bright plucks and airy pads create an uplifting mood. This track would fit well in a festival setting or a high-speed car chase in a video game.',
            # 'A dreamy ambient track featuring a soft pad melody and gentle synth arpeggios. A warm bassline adds depth while distant chimes and subtle white noise create an atmospheric, floating sensation. This piece could be playing in a meditation session or a futuristic sci-fi scene.',
            # 'A soft acoustic guitar gently strums a calming melody, accompanied by light percussion. A warm, deep bass softly hums in the background. The atmosphere is peaceful and relaxing, making it perfect for a quiet morning or a cozy café.',
            # 'A smooth jazz piece led by a gentle piano melody, accompanied by a double bass and soft brush drumming. A saxophone occasionally plays a few warm notes, adding to the relaxed atmosphere. This could be the perfect background music for a late-night coffee shop.',
            # 'A lively funk groove with a punchy bassline and rhythmic electric guitar strumming. The drums keep a steady beat, while a brass section occasionally joins in with energetic stabs. This track could be playing at a retro dance party.',
            # 'A high-energy rock track with distorted electric guitars and a driving drum beat. The rhythm is fast and powerful, with occasional breaks that add intensity. It feels like a song that could be playing in a road trip montage or an action-packed scene.',
            
            #demo
            'A soulful blues track with heartfelt electric guitar bends and a steady, laid-back groove. The warm tone of the guitar sings over a subtle rhythm section, evoking themes of longing and resilience. Perfect for a quiet bar scene or a contemplative evening drive.',
            'An anthemic rock track featuring gritty power chords, pounding drums, and a soaring lead guitar line. The energy is raw and rebellious, ideal for a stadium performance or a climactic movie showdown.',
            'A glitchy, futuristic electronic track with stuttering beats, evolving textures, and pulsating synth patterns. The track’s immersive sound design creates a sci-fi vibe, perfect for a cyberpunk chase scene or an underground rave.',
            'A vibrant house tune built on a deep kick drum, rhythmic claps, and catchy vocal chops. Bright synth stabs and filtered build-ups keep the momentum high, making it a great fit for a rooftop party or summer festival.',
            'A catchy pop anthem with shimmering synths, a punchy beat, and a memorable vocal hook. The upbeat mood and polished production make it ideal for a feel-good montage or a teen coming-of-age film soundtrack.',
            
            'A down-tempo electric blues track featuring expressive guitar bends, fingerpicked licks, and subtle vibrato that carries emotional weight. The rhythm section includes a brushed snare, softly thumping upright bass, and occasional organ swells that add warmth and depth. The atmosphere is raw yet intimate, perfect for a smoky bar at midnight or a montage of solitary reflection on a rainy night.',
            'A gritty hard rock anthem driven by heavily distorted guitars and thunderous drum patterns. The opening riff commands attention with a palm-muted chug, building into an explosive chorus with layered power chords and aggressive cymbal crashes. The bridge features a shredding guitar solo drenched in reverb, injecting intensity and drama. Ideal for a rebellious youth montage, an action movie chase, or a high-speed road trip sequence.',
            'An intricate electronic production blending tight, syncopated beats with glitchy effects and modular synth arpeggios. The low end pulses with a sub-heavy kick and sidechained bass, while shimmering textures and granular samples evolve gradually throughout the track. Occasional vocal chops and pitch-shifted risers create a futuristic, immersive soundscape, suited for a sci-fi game menu, high-tech commercial, or a VR dancefloor experience.',
            'A polished deep house track with a steady four-on-the-floor beat, accented by crisp hi-hats and reverb-soaked claps. A warm, round bassline grooves beneath ethereal synth chords, while filtered pads ebb and flow to create a dreamy progression. Mid-track, a melodic vocal loop enters with subtle delay and echo, adding a human touch to the electronic landscape. This track would be right at home in a sunset beach party, upscale rooftop lounge, or a fashion show runway.',
            'A vibrant modern pop track with glossy production, featuring a snappy kick-snare rhythm, layered synths, and infectious vocal hooks with tight harmonies. Verses are minimal and rhythmic, leading into a bright, anthemic chorus with rising chord progressions and punchy transitions. The bridge introduces a breakdown with airy pads and emotional vocal ad-libs before returning to a final chorus burst. Perfect for a coming-of-age film, a TikTok dance trend, or the closing scene of a romantic comedy.'
            ]

L = 2200
audio_num = 0
with torch.autocast(device_type="cuda", dtype=torch.float32):
    with torch.no_grad():
        device = 'cuda'
        for idx, i in enumerate(captions):
            for _ in range(1):
                # if os.path.isfile(os.path.join(save_path, f'{idx}_{audio_num}.npy')):
                #     continue
                description = [i] * 5
                # print(len(description), len(i))
                # break
                # print(len(i['ytid']))
                prompt_seq = model(description=description, length=L, g_scale=3)
                # print(prompt_seq.shape, len(description))

                for b in range(5):
                    np.save(os.path.join(save_path, f'{idx}_{audio_num}.npy'), prompt_seq[b, :, :L])
                    audio_num += 1
