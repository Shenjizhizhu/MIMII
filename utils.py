import os
import glob
import random
import toml
import json
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from typing import Optional,List,Dict,Tuple

class MIMIIToolkit:
    def __init__(self,config_path:str):
        self.config = toml.load(config_path)
        self.root_dir = self.config["basic"]["root_dir"]
        self.device_type = self.config["basic"]["device_type"]
        self.sample_rate = self.config["basic"]["sample_rate"]
        self.train_ratio = self.config["basic"]["train_ratio"]
        self.val_ratio = self.config["basic"]["val_ratio"]
        self.test_ratio = self.config["basic"]["test_ratio"]
        self.cross_validation = self.config["basic"]["cross_validation"]
        self.cv_folds = self.config["basic"]["cv_folds"]

        self.audio_duration = self.config["preprocess"]["audio_duration"]
        self.normalization = self.config["preprocess"]["normalization"]
        self.remove_silence = self.config["preprocess"]["remove_silence"]
        self.silence_threshold = self.config["preprocess"]["silence_threshold"]
        self.augmentations = self.config["preprocess"]["augmentations"]
        self.augment_methods = self.config["preprocess"]["augment_methods"]
        self.noise_strength = self.config["preprocess"]["noise_strength"]
        self.time_shift_step = self.config["preprocess"]["time_shift_step"]
        self.time_stretch_rate = self.config["preprocess"]["time_stretch_rate"]

        self.feature_type = self.config["feature"]["feature_type"]
        self.n_fft = self.config["feature"]["mel_spectrogram"]["n_fft"]
        self.hop_length = self.config["feature"]["mel_spectrogram"]["hop_length"]
        self.n_mels = self.config["feature"]["mel_spectrogram"]["n_mels"]
        self.feature_normalize = self.config["feature"]["feature_normalize"]
        self.log_scale = self.config["feature"]["log_scale"]

        self.epochs = self.config["loader"]["epochs"]
        self.lr = self.config["loader"]["lr"]
        self.lr_scheduler = self.config["loader"]["lr_scheduler"]
        self.step_size = self.config["loader"]["step_size"]
        self.gamma = self.config["loader"]["gamma"]
        self.weight_decay = self.config["loader"]["weight_decay"]
        self.early_stopping = self.config["loader"]["early_stopping"]
        self.patience = self.config["loader"]["patience"]
        self.save_dir = self.config["loader"]["save_dir"]
        self.log_dir = self.config["loader"]["log_dir"]

        self.metrics = self.config["testing"]["metrics"]
        self.result_dir = self.config["testing"]["result_dir"]
        self.plot_curves = self.config["testing"]["plot_curves"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "训练/验证/测试比例总和必须为1（当前：{}+{}+{}={}）".format(
                self.train_ratio, self.val_ratio, self.test_ratio,
                self.train_ratio + self.val_ratio + self.test_ratio
            )

    def _remove_silence(self,audio_data:np.ndarray) -> np.ndarray:
        if not self.remove_silence:
            return audio_data
        energy = np.abs(audio_data)
        non_silent_frames = np.where(energy > self.silence_threshold)[0]
        if len(non_silent_frames) == 0:
            return audio_data
        start = non_silent_frames[0] * self.hop_length
        end = non_silent_frames[-1] * self.hop_length + 1
        audio_data = audio_data[start:end]
        target_frames = int(self.audio_duration * self.sample_rate)
        if len(audio_data) > target_frames:
            audio_data = audio_data[:target_frames]
        else:
            audio_data = audio_data = np.pad(audio_data,(0,target_frames - len(audio_data)))    
        return audio_data
    
    def _normalize_audio(self,audio_data:np.ndarray) -> np.ndarray:
        if not self.normalization:
            return audio_data
        if self.normalization == "zscore":
            audio_data = (audio_data - np.mean(audio_data)) / (audio_data.std() + 1e-8)
        elif self.normalization == "minmax":
            audio_data = (audio_data - audio_data.min()) / (audio_data.max() - audio_data.min() + 1e-8)
        return audio_data
        
    
    def load_audio(self,wav_path:str,dtype: str = "float64") -> Tuple[np.ndarray,int]:
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"音频文件不存在:{wav_path}")
        data,sr = sf.read(wav_path,dtype =dtype)
        if len(data.shape) > 1:
            data = np.mean(data,axis = 1)
        if sr != self.sample_rate:
            data = librosa.resample(data,orig_sr = sr,target_sr = self.sample_rate)
            sr = self.sample_rate
        target_frames = int(self.audio_duration * self.sample_rate)
        if len(data) > target_frames:
            data = data[:target_frames]         
        else:    
            data = np.pad(data,(0,target_frames - len(data)),mode = 'constant')
        if self.remove_silence:
            data = self._remove_silence(data)
        data = self._normalize_audio(data)
        return data,sr

    def save_audio(self,save_path:str,audio_data:np.ndarray,subtype:str = "PCM_16") -> None:
        os.makedirs(os.path.dirname(save_path),exist_ok = True)
        if audio_data.dtype in[np.float32,np.float64]:
            audio_data = np.clip(audio_data,-1.0,1.0)
        elif audio_data.dtype == np.int16:
            audio_data = np.clip(audio_data,-32768,32767)
        sf.write(file=save_path,data=audio_data,samplerate=self.sample_rate,format="WAV",subtype=subtype)
    
    def audio_augment(self,audio_data:np.ndarray) -> np.ndarray:
        if not self.augmentations:
            return audio_data
        aug_data = audio_data.copy()
        sr = self.sample_rate
        if "noise" in self.augment_methods:
            noise = np.random.randn(len(aug_data)) * self.noise_strength
            aug_data = aug_data + noise
            aug_data = np.clip(aug_data,-1.0,1.0)
        if "time_shift" in self.augment_methods:
            shift_frames = int(self.time_shift_step * sr)
            direction = random.choice([-1,1])
            aug_data = np.roll(aug_data,int(direction * shift_frames))
        if "pitch_shift" in self.augment_methods:
            n_steps = random.uniform(-self.time_shift_step,self.time_shift_step)
            aug_data = librosa.effects.pitch_shift(aug_data,sr=sr,n_steps=n_steps)
        if "time_stretch" in self.augment_methods:
            aug_data = librosa.effects.time_stretch(aug_data,rate = self.time_stretch_rate)
            target_frames = int(self.audio_duration * sr)
            if len(aug_data) < target_frames:
                aug_data = aug_data[:target_frames]
            else:
                aug_data = np.pad(aug_data,(0,target_frames - len(aug_data)),mode="constant")
        return aug_data
    
    def wav2mel(self, audio_data: np.ndarray, use_torchaudio: bool = False) -> np.ndarray:
        if self.feature_type != "mel_spectrogram":
            raise ValueError(f"仅支持mel_spectrogram特征，当前指定：{self.feature_type}")
        
        if len(audio_data.shape) != 1:
            raise ValueError(f"音频数据必须是一维数组，当前形状：{audio_data.shape}")
        
        sr = self.sample_rate
        n_fft = self.n_fft
        hop_length = self.hop_length
        n_mels = self.n_mels

        if use_torchaudio:
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            mel_transform = T.MelSpectrogram(sample_rate=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,f_min=0.0,f_max=8000.0,power=2.0,normalized=False).to(self.device)

            mel_spec = mel_transform(waveform)
            
            if self.log_scale:
                mel_spec = torch.clamp(mel_spec, min=1e-10)
                mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, ref=1.0,amin=1e-10,top_db=80.0)
            else:
                mel_spec_db = mel_spec

            mel_spec_db = mel_spec_db.squeeze(0).squeeze(0).cpu().numpy()
        
        else:
            mel_spec = librosa.feature.melspectrogram(y=audio_data,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,fmin=0.0,fmax=8000.0,power=2.0,norm='slaney')

            if self.log_scale:
                mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0,amin=1e-10,top_db=80.0)
            else:
                mel_spec_db = mel_spec

        if self.feature_normalize:
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        mel_spec_db = mel_spec_db.astype(np.float32)
        
        return mel_spec_db

    def _split_dataset(self,all_metadata:List[Dict]) -> Dict[str,List[Dict]]:
        train_metadata,temp_metadata = train_test_split(all_metadata,train_size = self.train_ratio,random_state = 42,stratify=[f"{m['device_type']}_{m['device_id']}_{m['label']}" for m in all_metadata])
        val_test_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_metadata,test_metadata = train_test_split(temp_metadata,train_size = val_test_ratio,random_state = 42,stratify=[f"{m['device_type']}_{m['device_id']}_{m['label']}" for m in temp_metadata])      
        return {"train":train_metadata,"val":val_metadata,"test":test_metadata}
    
    def batch_process_mimii(self) -> Dict[str,List[Dict]]:
        os.makedirs(self.save_dir,exist_ok = True)
        os.makedirs(self.log_dir,exist_ok = True)
        os.makedirs(self.result_dir,exist_ok = True)

        label_map = {"normal":0,"abnormal":1}
        all_metadata = []

        for dev_type in self.device_type:
            wav_pattern = os.path.join(self.root_dir,dev_type,"**","*.wav")
            wav_paths = glob.glob(wav_pattern,recursive = True)
            if not wav_paths:
                print(f"警告：设备{dev_type}未找到任何音频文件")
                continue
            for wav_path in tqdm(wav_paths,desc = f"Processing{dev_type}"):
                rel_path = os.path.relpath(wav_path,self.root_dir)
                parts = rel_path.split(os.sep)
                if len(parts) < 4:
                    continue
                device_id = parts[1]
                status = parts[2]
                filename = parts[3]
                base_name = os.path.splitext(filename)[0]

                audio_data,_ = self.load_audio(wav_path)
                if self.augmentations:
                    audio_data = self.audio_augment(audio_data)
                    aug_audio_path = os.path.join(self.save_dir,"aug_audio",dev_type,device_id,status,f"{base_name}_aug.wav")
                    self.save_audio(aug_audio_path,audio_data)
                mel_spec = self.wav2mel(audio_data,use_torchaudio = False)
                mel_save_path = os.path.join(self.save_dir,"mel_spectrogram",dev_type,device_id,status,f"{base_name}.npy")
                os.makedirs(os.path.dirname(mel_save_path),exist_ok = True)
                np.save(mel_save_path,mel_spec)
                metadata = {"mel_path":mel_save_path,"audio_path":wav_path,"device_type":dev_type,"device_id":device_id,"status":status,"label":label_map[status],"filename":filename}
                all_metadata.append(metadata)

            if self.cross_validation:
                kf = KFold(n_splits = self.cv_folds,shuffle = True,random_state = 42)
                cv_splits = []
                for fold,(train_idx,val_idx) in enumerate(kf.split(all_metadata)):
                    cv_splits.append({"fold":fold,"train":[all_metadata[i] for i in train_idx],"val":[all_metadata[i] for i in val_idx]})
                    with open(os.path.join(self.save_dir,"cv_splits.json"),"w",encoding = "utf-8") as f:
                        json.dump(cv_splits,f,indent = 4)
                        split_metadata = {"cv_splits":cv_splits}
            else:
                split_metadata = self._split_dataset(all_metadata)
                with open(os.path.join(self.save_dir, "all_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(all_metadata, f, indent=4)
                with open(os.path.join(self.save_dir, "train_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(split_metadata["train"], f, indent=4)
                with open(os.path.join(self.save_dir, "val_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(split_metadata["val"], f, indent=4)
                with open(os.path.join(self.save_dir, "test_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(split_metadata["test"], f, indent=4)

                with open(os.path.join(self.save_dir,"config_copy.toml"),"w",encoding="utf-8") as f:
                    toml.dump(self.config,f)
                print(f"\n处理完成！")
                print(f"全量样本数: {len(all_metadata)}")

                if not self.cross_validation:
                    print(f"训练集样本数: {len(split_metadata['train'])}")
                    print(f"验证集样本数: {len(split_metadata['val'])}")
                    print(f"测试集样本数: {len(split_metadata['test'])}")
                else:
                    print(f"交叉验证折数: {self.cv_folds}")

                return split_metadata
    def load_mel_from_metadata(self,metadata_type:str = "train",dev_type:Optional[str] = None,label:Optional[int] = None,max_samples:Optional[int] = None) -> Tuple[np.ndarray,np.ndarray]:
        metadata_path = os.path.join(self.save_dir,f"{metadata_type}_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在:{metadata_path}")
        with open(metadata_path,"r",encoding = "uf-8") as f:
            metadata = toml.load(f)
            if dev_type is not None:
                metadata = [m for m in metadata if m["device_type"] == dev_type]
            if label is not None:
                metadata = [m for m in metadata if m["label"] == label]
            if max_samples is not None:
                metadata = metadata[:max_samples]
            mel_data = []
            labels = []
        for m in tqdm(metadata,desc = f"Loading{metadata_type} mel spectrograms"):
            mel_spec = np.load(m["mel_path"])
            mel_data.append(mel_spec)
            labels.append(m["label"])
        return np.array(mel_data),np.array(labels)
    
if __name__ == "__main__":
    toolkit = MIMIIToolkit(config_path="./mimii_config.toml")
    split_metadata = toolkit.batch_process_mimii()
    train_mel, train_labels = toolkit.load_mel_from_metadata(metadata_type="train",dev_type="fan",label=0,max_samples=100)
    
    print(f"\n训练集梅尔频谱形状: {train_mel.shape}")
    print(f"训练集标签分布: {np.bincount(train_labels)}")
    
