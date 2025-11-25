import React from 'react';

interface DataTableProps {
  data: Record<string, unknown>[];
  maxRows?: number;
}

const DataTable: React.FC<DataTableProps> = ({ data, maxRows = 10 }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        Nu există date de afișat
      </div>
    );
  }

  const columns = Object.keys(data[0]);
  const displayData = data.slice(0, maxRows);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              #
            </th>
            {columns.map((col) => (
              <th
                key={col}
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {displayData.map((row, rowIndex) => (
            <tr key={rowIndex} className="hover:bg-gray-50">
              <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                {rowIndex + 1}
              </td>
              {columns.map((col) => (
                <td
                  key={col}
                  className="px-4 py-3 whitespace-nowrap text-sm text-gray-900"
                >
                  {typeof row[col] === 'number'
                    ? (row[col] as number).toFixed(4)
                    : String(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.length > maxRows && (
        <div className="text-center text-gray-500 py-3 text-sm">
          Afișate {maxRows} din {data.length} rânduri
        </div>
      )}
    </div>
  );
};

export default DataTable;
