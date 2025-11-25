import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface TrainingChartProps {
  history: {
    train_loss: number[];
    val_loss: number[];
    train_acc: number[];
    val_acc: number[];
  };
}

const TrainingChart: React.FC<TrainingChartProps> = ({ history }) => {
  // Prepare data for charts
  const lossData = history.train_loss.map((loss, index) => ({
    epoch: index + 1,
    'Train Loss': loss,
    'Val Loss': history.val_loss[index] || null,
  }));

  const accData = history.train_acc.map((acc, index) => ({
    epoch: index + 1,
    'Train Accuracy': acc * 100,
    'Val Accuracy': history.val_acc[index] ? history.val_acc[index] * 100 : null,
  }));

  return (
    <div className="space-y-8">
      {/* Loss Chart */}
      <div className="bg-white rounded-lg p-4 shadow">
        <h3 className="text-lg font-semibold mb-4 text-gray-700">Loss în Timp</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={lossData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" label={{ value: 'Epoca', position: 'insideBottom', offset: -5 }} />
            <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="Train Loss"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="Val Loss"
              stroke="#EF4444"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy Chart */}
      <div className="bg-white rounded-lg p-4 shadow">
        <h3 className="text-lg font-semibold mb-4 text-gray-700">Acuratețe în Timp</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={accData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" label={{ value: 'Epoca', position: 'insideBottom', offset: -5 }} />
            <YAxis
              label={{ value: 'Acuratețe (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 100]}
            />
            <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
            <Legend />
            <Line
              type="monotone"
              dataKey="Train Accuracy"
              stroke="#10B981"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="Val Accuracy"
              stroke="#F59E0B"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TrainingChart;
