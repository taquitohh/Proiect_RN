import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
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
  getStatus: () => api.get<StatusResponse>('/api/status'),

  // Data
  uploadData: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post<{ message: string; info: DataInfo }>('/api/data/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  generateData: (params: SyntheticDataRequest) =>
    api.post<{ message: string; info: DataInfo }>('/api/data/generate', params),

  getDataInfo: () => api.get<DataInfo>('/api/data/info'),

  getStatistics: () => api.get('/api/data/statistics'),

  // Preprocessing
  preprocess: (params: PreprocessRequest) =>
    api.post<PreprocessResponse>('/api/preprocess', params),

  getAnalysis: () => api.get('/api/preprocess/analysis'),

  // Training
  train: (params: TrainRequest) => api.post<TrainResponse>('/api/train', params),

  getTrainingHistory: () =>
    api.get<{ train_loss: number[]; val_loss: number[]; train_acc: number[]; val_acc: number[] }>(
      '/api/train/history'
    ),

  evaluate: () =>
    api.get<{ test_loss: number; accuracy: number; n_samples: number }>('/api/train/evaluate'),

  // Prediction
  predict: (params: PredictRequest) => api.post<PredictResponse>('/api/predict', params),
};

export default api;
