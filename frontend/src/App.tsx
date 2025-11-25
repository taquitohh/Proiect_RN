import { useState, useEffect } from 'react';
import {
  Database,
  Settings,
  Play,
  Brain,
  BarChart3,
  CheckCircle,
  XCircle,
  RefreshCw,
} from 'lucide-react';
import { apiService, DataInfo, TrainResponse, StatusResponse } from './api';
import { FileUpload, TrainingChart, DataTable, LoadingSpinner } from './components';

type TabType = 'data' | 'preprocess' | 'train' | 'results';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('data');
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [dataInfo, setDataInfo] = useState<DataInfo | null>(null);
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  // Form states
  const [syntheticParams, setSyntheticParams] = useState({
    n_samples: 1000,
    n_features: 10,
    n_classes: 3,
    noise: 0.1,
  });

  const [preprocessParams, setPreprocessParams] = useState({
    target_column: 'target',
    train_ratio: 0.8,
    validation_ratio: 0.1,
    test_ratio: 0.1,
    normalize: true,
    handle_missing: true,
  });

  const [trainParams, setTrainParams] = useState({
    hidden_layers: [128, 64, 32],
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    dropout: 0.2,
  });

  // Load status on mount
  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await apiService.getStatus();
      setStatus(response.data);
    } catch {
      console.log('API not available');
    }
  };

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.uploadData(file);
      setDataInfo(response.data.info);
      setMessage(response.data.message);
      fetchStatus();
    } catch (err) {
      setError('Eroare la încărcarea fișierului');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.generateData(syntheticParams);
      setDataInfo(response.data.info);
      setMessage(response.data.message);
      fetchStatus();
    } catch (err) {
      setError('Eroare la generarea datelor');
    } finally {
      setLoading(false);
    }
  };

  const handlePreprocess = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.preprocess(preprocessParams);
      setMessage(
        `${response.data.message} - Train: ${response.data.train_size}, Val: ${response.data.validation_size}, Test: ${response.data.test_size}`
      );
      fetchStatus();
    } catch (err) {
      setError('Eroare la preprocesare');
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.train(trainParams);
      setTrainResult(response.data);
      setMessage(response.data.message);
      fetchStatus();
    } catch (err) {
      setError('Eroare la antrenare');
    } finally {
      setLoading(false);
    }
  };

  const StatusIndicator = ({ active, label }: { active: boolean; label: string }) => (
    <div className="flex items-center space-x-2">
      {active ? (
        <CheckCircle className="w-5 h-5 text-green-500" />
      ) : (
        <XCircle className="w-5 h-5 text-gray-300" />
      )}
      <span className={active ? 'text-green-600' : 'text-gray-400'}>{label}</span>
    </div>
  );

  const tabs = [
    { id: 'data' as TabType, label: 'Date', icon: Database },
    { id: 'preprocess' as TabType, label: 'Preprocesare', icon: Settings },
    { id: 'train' as TabType, label: 'Antrenare', icon: Brain },
    { id: 'results' as TabType, label: 'Rezultate', icon: BarChart3 },
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Brain className="w-8 h-8" />
              <div>
                <h1 className="text-2xl font-bold">Neural Network Dashboard</h1>
                <p className="text-blue-100 text-sm">Proiect Rețele Neuronale - POLITEHNICA București</p>
              </div>
            </div>
            <button
              onClick={fetchStatus}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              title="Reîncarcă status"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Status Bar */}
      <div className="bg-white border-b shadow-sm">
        <div className="container mx-auto px-6 py-3">
          <div className="flex items-center space-x-8">
            <StatusIndicator active={status?.data_loaded || false} label="Date încărcate" />
            <StatusIndicator active={status?.preprocessed || false} label="Date preprocesate" />
            <StatusIndicator active={status?.model_trained || false} label="Model antrenat" />
            {status?.data_shape && (
              <span className="text-gray-500 text-sm">
                Shape: {status.data_shape[0]} × {status.data_shape[1]}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="container mx-auto px-6">
          <nav className="flex space-x-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center space-x-2 px-6 py-4 border-b-2 transition-colors
                  ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                <tab.icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Messages */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}
        {message && (
          <div className="mb-6 bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg">
            {message}
          </div>
        )}

        {loading && <LoadingSpinner message="Se procesează..." />}

        {!loading && (
          <>
            {/* Data Tab */}
            {activeTab === 'data' && (
              <div className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Upload */}
                  <div className="card">
                    <h2 className="text-xl font-semibold mb-4 text-gray-700">Încarcă Date</h2>
                    <FileUpload onUpload={handleFileUpload} accept=".csv" />
                  </div>

                  {/* Generate */}
                  <div className="card">
                    <h2 className="text-xl font-semibold mb-4 text-gray-700">Generează Date Sintetice</h2>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-600 mb-1">
                            Număr eșantioane
                          </label>
                          <input
                            type="number"
                            value={syntheticParams.n_samples}
                            onChange={(e) =>
                              setSyntheticParams({ ...syntheticParams, n_samples: parseInt(e.target.value) })
                            }
                            className="input-field"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-600 mb-1">
                            Număr caracteristici
                          </label>
                          <input
                            type="number"
                            value={syntheticParams.n_features}
                            onChange={(e) =>
                              setSyntheticParams({ ...syntheticParams, n_features: parseInt(e.target.value) })
                            }
                            className="input-field"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-600 mb-1">Număr clase</label>
                          <input
                            type="number"
                            value={syntheticParams.n_classes}
                            onChange={(e) =>
                              setSyntheticParams({ ...syntheticParams, n_classes: parseInt(e.target.value) })
                            }
                            className="input-field"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-600 mb-1">Zgomot</label>
                          <input
                            type="number"
                            step="0.01"
                            value={syntheticParams.noise}
                            onChange={(e) =>
                              setSyntheticParams({ ...syntheticParams, noise: parseFloat(e.target.value) })
                            }
                            className="input-field"
                          />
                        </div>
                      </div>
                      <button onClick={handleGenerateData} className="btn-primary w-full">
                        <Play className="w-4 h-4 inline mr-2" />
                        Generează Date
                      </button>
                    </div>
                  </div>
                </div>

                {/* Data Preview */}
                {dataInfo && (
                  <div className="card">
                    <h2 className="text-xl font-semibold mb-4 text-gray-700">Previzualizare Date</h2>
                    <div className="grid md:grid-cols-3 gap-4 mb-6">
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{dataInfo.n_samples}</div>
                        <div className="text-sm text-gray-600">Eșantioane</div>
                      </div>
                      <div className="bg-green-50 p-4 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{dataInfo.n_features}</div>
                        <div className="text-sm text-gray-600">Caracteristici</div>
                      </div>
                      <div className="bg-purple-50 p-4 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          {dataInfo.memory_usage?.toFixed(2)} MB
                        </div>
                        <div className="text-sm text-gray-600">Memorie</div>
                      </div>
                    </div>
                    {dataInfo.preview && <DataTable data={dataInfo.preview} maxRows={10} />}
                  </div>
                )}
              </div>
            )}

            {/* Preprocess Tab */}
            {activeTab === 'preprocess' && (
              <div className="card max-w-2xl mx-auto">
                <h2 className="text-xl font-semibold mb-6 text-gray-700">Configurare Preprocesare</h2>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-600 mb-1">Coloana țintă</label>
                    <input
                      type="text"
                      value={preprocessParams.target_column}
                      onChange={(e) =>
                        setPreprocessParams({ ...preprocessParams, target_column: e.target.value })
                      }
                      className="input-field"
                    />
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-1">Train %</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={preprocessParams.train_ratio}
                        onChange={(e) =>
                          setPreprocessParams({ ...preprocessParams, train_ratio: parseFloat(e.target.value) })
                        }
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-1">Validation %</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={preprocessParams.validation_ratio}
                        onChange={(e) =>
                          setPreprocessParams({
                            ...preprocessParams,
                            validation_ratio: parseFloat(e.target.value),
                          })
                        }
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-1">Test %</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={preprocessParams.test_ratio}
                        onChange={(e) =>
                          setPreprocessParams({ ...preprocessParams, test_ratio: parseFloat(e.target.value) })
                        }
                        className="input-field"
                      />
                    </div>
                  </div>

                  <div className="flex items-center space-x-6">
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={preprocessParams.normalize}
                        onChange={(e) =>
                          setPreprocessParams({ ...preprocessParams, normalize: e.target.checked })
                        }
                        className="w-4 h-4 text-blue-600 rounded"
                      />
                      <span className="text-gray-700">Normalizare</span>
                    </label>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={preprocessParams.handle_missing}
                        onChange={(e) =>
                          setPreprocessParams({ ...preprocessParams, handle_missing: e.target.checked })
                        }
                        className="w-4 h-4 text-blue-600 rounded"
                      />
                      <span className="text-gray-700">Tratare valori lipsă</span>
                    </label>
                  </div>

                  <button
                    onClick={handlePreprocess}
                    disabled={!status?.data_loaded}
                    className="btn-success w-full disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Settings className="w-4 h-4 inline mr-2" />
                    Preprocesează Date
                  </button>
                </div>
              </div>
            )}

            {/* Train Tab */}
            {activeTab === 'train' && (
              <div className="card max-w-2xl mx-auto">
                <h2 className="text-xl font-semibold mb-6 text-gray-700">Configurare Antrenare</h2>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-600 mb-1">
                      Straturi ascunse (separate prin virgulă)
                    </label>
                    <input
                      type="text"
                      value={trainParams.hidden_layers.join(', ')}
                      onChange={(e) =>
                        setTrainParams({
                          ...trainParams,
                          hidden_layers: e.target.value.split(',').map((s) => parseInt(s.trim()) || 64),
                        })
                      }
                      className="input-field"
                      placeholder="128, 64, 32"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-1">Epoci</label>
                      <input
                        type="number"
                        value={trainParams.epochs}
                        onChange={(e) =>
                          setTrainParams({ ...trainParams, epochs: parseInt(e.target.value) })
                        }
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-1">Batch Size</label>
                      <input
                        type="number"
                        value={trainParams.batch_size}
                        onChange={(e) =>
                          setTrainParams({ ...trainParams, batch_size: parseInt(e.target.value) })
                        }
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-1">Learning Rate</label>
                      <input
                        type="number"
                        step="0.0001"
                        value={trainParams.learning_rate}
                        onChange={(e) =>
                          setTrainParams({ ...trainParams, learning_rate: parseFloat(e.target.value) })
                        }
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-1">Dropout</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={trainParams.dropout}
                        onChange={(e) =>
                          setTrainParams({ ...trainParams, dropout: parseFloat(e.target.value) })
                        }
                        className="input-field"
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleTrain}
                    disabled={!status?.preprocessed}
                    className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Brain className="w-4 h-4 inline mr-2" />
                    Antrenează Model
                  </button>
                </div>
              </div>
            )}

            {/* Results Tab */}
            {activeTab === 'results' && (
              <div className="space-y-6">
                {trainResult ? (
                  <>
                    {/* Metrics Cards */}
                    <div className="grid md:grid-cols-4 gap-4">
                      <div className="card text-center">
                        <div className="text-3xl font-bold text-blue-600">
                          {trainResult.epochs_trained}
                        </div>
                        <div className="text-sm text-gray-600">Epoci antrenate</div>
                      </div>
                      <div className="card text-center">
                        <div className="text-3xl font-bold text-green-600">
                          {(trainResult.final_train_acc * 100).toFixed(2)}%
                        </div>
                        <div className="text-sm text-gray-600">Acuratețe Train</div>
                      </div>
                      <div className="card text-center">
                        <div className="text-3xl font-bold text-orange-600">
                          {trainResult.final_val_acc ? (trainResult.final_val_acc * 100).toFixed(2) : 'N/A'}%
                        </div>
                        <div className="text-sm text-gray-600">Acuratețe Validare</div>
                      </div>
                      <div className="card text-center">
                        <div className="text-3xl font-bold text-purple-600">
                          {trainResult.final_train_loss.toFixed(4)}
                        </div>
                        <div className="text-sm text-gray-600">Loss Final</div>
                      </div>
                    </div>

                    {/* Training Charts */}
                    <div className="card">
                      <h2 className="text-xl font-semibold mb-4 text-gray-700">Grafice Antrenare</h2>
                      <TrainingChart history={trainResult.history} />
                    </div>
                  </>
                ) : (
                  <div className="card text-center py-12">
                    <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-xl font-medium text-gray-500">Nu există rezultate</h3>
                    <p className="text-gray-400 mt-2">
                      Antrenează un model pentru a vedea rezultatele aici.
                    </p>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 py-4 mt-8">
        <div className="container mx-auto px-6 text-center text-sm">
          <p>Proiect Rețele Neuronale - POLITEHNICA București - FIIR</p>
          <p className="mt-1">Etapa 3: Analiza și Pregătirea Setului de Date</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
