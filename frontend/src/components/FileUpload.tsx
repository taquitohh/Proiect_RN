import React, { useCallback, useState } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';

interface FileUploadProps {
  onUpload: (file: File) => void;
  accept?: string;
  maxSize?: number;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onUpload,
  accept = '.csv',
  maxSize = 50 * 1024 * 1024, // 50MB
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const validateFile = (file: File): boolean => {
    if (file.size > maxSize) {
      setError(`Fișierul este prea mare. Maxim ${maxSize / 1024 / 1024}MB.`);
      return false;
    }

    const extension = file.name.split('.').pop()?.toLowerCase();
    const acceptedExtensions = accept.split(',').map((ext) => ext.trim().replace('.', ''));

    if (!acceptedExtensions.includes(extension || '')) {
      setError(`Tip de fișier invalid. Acceptat: ${accept}`);
      return false;
    }

    return true;
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      setError(null);

      const file = e.dataTransfer.files[0];
      if (file && validateFile(file)) {
        setFileName(file.name);
        onUpload(file);
      }
    },
    [onUpload]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const file = e.target.files?.[0];
    if (file && validateFile(file)) {
      setFileName(file.name);
      onUpload(file);
    }
  };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${error ? 'border-red-300 bg-red-50' : ''}
        `}
      >
        <input
          type="file"
          accept={accept}
          onChange={handleFileChange}
          className="hidden"
          id="file-upload"
        />
        <label htmlFor="file-upload" className="cursor-pointer">
          <div className="flex flex-col items-center">
            {fileName ? (
              <>
                <FileText className="w-12 h-12 text-green-500 mb-3" />
                <p className="text-lg font-medium text-gray-700">{fileName}</p>
                <p className="text-sm text-gray-500 mt-1">Fișier selectat</p>
              </>
            ) : (
              <>
                <Upload className="w-12 h-12 text-gray-400 mb-3" />
                <p className="text-lg font-medium text-gray-700">
                  Trage fișierul aici sau click pentru încărcare
                </p>
                <p className="text-sm text-gray-500 mt-1">Acceptat: {accept}</p>
              </>
            )}
          </div>
        </label>
      </div>

      {error && (
        <div className="mt-3 flex items-center text-red-600">
          <AlertCircle className="w-4 h-4 mr-2" />
          <span className="text-sm">{error}</span>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
