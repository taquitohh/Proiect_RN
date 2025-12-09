import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface DataInfo {
  n_samples: number;
  n_features: number;
  columns: string[];
  dtypes: Record<string, string>;
  missing_values: Record<string, number>;
  memory_usage: number;
  preview?: Record<string, unknown>[];
}

export interface SyntheticDataRequest {
  n_samples: number;
  n_features: number;
  n_classes: number;
  noise: number;
}

export interface PreprocessRequest {
  target_column: string;
  train_ratio: number;
  validation_ratio: number;
  test_ratio: number;
  normalize: boolean;
  handle_missing: boolean;
}

export interface PreprocessResponse {
  message: string;
  train_size: number;
  validation_size: number;
  test_size: number;
  analysis: {
    shape: [number, number];
    issues: string[];
  };
}

export interface TrainRequest {
  hidden_layers: number[];
  epochs: number;
  batch_size: number;
  learning_rate: number;
  dropout: number;
}

export interface TrainResponse {
  message: string;
  epochs_trained: number;
  final_train_loss: number;
  final_val_loss: number | null;
  final_train_acc: number;
  final_val_acc: number | null;
  history: {
    train_loss: number[];
    val_loss: number[];
    train_acc: number[];
    val_acc: number[];
  };
}

export interface StatusResponse {
  data_loaded: boolean;
  data_shape: [number, number] | null;
  preprocessed: boolean;
  model_trained: boolean;
}

export interface PredictRequest {
  data: number[][];
}

export interface PredictResponse {
  predictions: number[];
  probabilities: number[][];
}

// API Functions
export const apiService = {
  // Status
  getStatus: () => axiosInstance.get<StatusResponse>('/api/status'),

  // Data
  uploadData: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return axiosInstance.post<{ message: string; info: DataInfo }>('/api/data/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  generateData: (params: SyntheticDataRequest) =>
    axiosInstance.post<{ message: string; info: DataInfo }>('/api/data/generate', params),

  getDataInfo: () => axiosInstance.get<DataInfo>('/api/data/info'),

  getStatistics: () => axiosInstance.get('/api/data/statistics'),

  // Preprocessing
  preprocess: (params: PreprocessRequest) =>
    axiosInstance.post<PreprocessResponse>('/api/preprocess', params),

  getAnalysis: () => axiosInstance.get('/api/preprocess/analysis'),

  // Training
  train: (params: TrainRequest) => axiosInstance.post<TrainResponse>('/api/train', params),

  getTrainingHistory: () =>
    axiosInstance.get<{ train_loss: number[]; val_loss: number[]; train_acc: number[]; val_acc: number[] }>(
      '/api/train/history'
    ),

  evaluate: () =>
    axiosInstance.get<{ test_loss: number; accuracy: number; n_samples: number }>('/api/train/evaluate'),

  // Prediction
  predict: (params: PredictRequest) => axiosInstance.post<PredictResponse>('/api/predict', params),

  // Blender Text-to-Code
  generateBlenderCode: (text: string) => 
    axiosInstance.post<{
      success: boolean;
      intent: string;
      params: Record<string, number>;
      interpretation: string;
      code: string;
    }>('/api/blender/generate', { text }),

  getBlenderIntents: () => 
    axiosInstance.get<{
      intents: string[];
      keywords: Record<string, string[]>;
    }>('/api/blender/intents'),
};

export default apiService;
