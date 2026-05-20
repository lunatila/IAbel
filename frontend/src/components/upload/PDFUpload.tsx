import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, X, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { apiService } from '../../services/api';
import { UploadResponse, ReindexUpdate } from '../../types';
import Button from '../ui/Button';
import Card from '../ui/Card';
import Badge from '../ui/Badge';
import toast from 'react-hot-toast';

interface UploadedFile {
  file: File;
  status: 'uploading' | 'processing' | 'success' | 'error';
  progress: number;
  response?: UploadResponse;
  error?: string;
}

const PDFUpload: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isReindexing, setIsReindexing] = useState(false);
  const [reindexProgress, setReindexProgress] = useState<ReindexUpdate[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (files: FileList | null) => {
    if (!files) return;

    const pdfFiles = Array.from(files).filter(file => 
      file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
    );

    if (pdfFiles.length === 0) {
      toast.error('Por favor, selecione apenas arquivos PDF');
      return;
    }

    if (pdfFiles.length !== files.length) {
      toast.warning('Apenas arquivos PDF foram selecionados');
    }

    pdfFiles.forEach(file => {
      const uploadFile: UploadedFile = {
        file,
        status: 'uploading',
        progress: 0
      };

      setUploadedFiles(prev => [...prev, uploadFile]);
      uploadPDF(uploadFile);
    });
  };

  const uploadPDF = async (uploadFile: UploadedFile) => {
    try {
      const response = await apiService.uploadPDF(
        uploadFile.file,
        (progress) => {
          setUploadedFiles(prev =>
            prev.map(f =>
              f.file === uploadFile.file
                ? { ...f, progress, status: progress === 100 ? 'processing' : 'uploading' }
                : f
            )
          );
        }
      );

      setUploadedFiles(prev =>
        prev.map(f =>
          f.file === uploadFile.file
            ? { ...f, status: 'success', progress: 100, response }
            : f
        )
      );

      toast.success(`PDF "${uploadFile.file.name}" carregado com sucesso!`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Erro desconhecido';
      
      setUploadedFiles(prev =>
        prev.map(f =>
          f.file === uploadFile.file
            ? { ...f, status: 'error', error: errorMessage }
            : f
        )
      );

      toast.error(`Erro ao carregar "${uploadFile.file.name}": ${errorMessage}`);
    }
  };

  const removeFile = (fileToRemove: File) => {
    setUploadedFiles(prev => prev.filter(f => f.file !== fileToRemove));
  };

  const retryUpload = (uploadFile: UploadedFile) => {
    setUploadedFiles(prev =>
      prev.map(f =>
        f.file === uploadFile.file
          ? { ...f, status: 'uploading', progress: 0, error: undefined }
          : f
      )
    );
    uploadPDF(uploadFile);
  };

  const handleReindex = async (forceReindex: boolean = false) => {
    setIsReindexing(true);
    setReindexProgress([]);

    try {
      await apiService.reindexDocuments(forceReindex, (update) => {
        setReindexProgress(prev => [...prev, update]);
        
        if (update.type === 'complete') {
          toast.success('Reindexação concluída com sucesso!');
          setIsReindexing(false);
        } else if (update.type === 'error') {
          toast.error(`Erro na reindexação: ${update.error}`);
          setIsReindexing(false);
        }
      });
    } catch (error) {
      console.error('Reindex error:', error);
      toast.error('Erro durante a reindexação');
      setIsReindexing(false);
    }
  };

  const clearCompleted = () => {
    setUploadedFiles(prev => prev.filter(f => f.status !== 'success'));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle size={16} className="text-green-600" />;
      case 'error':
        return <AlertCircle size={16} className="text-red-600" />;
      default:
        return <FileText size={16} className="text-blue-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'error':
        return 'error';
      case 'processing':
        return 'warning';
      default:
        return 'info';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Upload de Documentos
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Carregue PDFs de engenharia de reservatórios para análise
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={clearCompleted}
            disabled={!uploadedFiles.some(f => f.status === 'success')}
          >
            Limpar Concluídos
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => handleReindex(false)}
            disabled={isReindexing}
            isLoading={isReindexing}
            icon={!isReindexing && <RefreshCw size={14} />}
          >
            Reindexar
          </Button>
        </div>
      </div>

      {/* Upload Area */}
      <Card
        className={`transition-all duration-200 ${
          isDragOver
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-dashed border-gray-300 dark:border-gray-600'
        }`}
      >
        <div
          className="text-center py-8"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload size={48} className="mx-auto text-gray-400 mb-4" />
          
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Arraste PDFs aqui ou clique para selecionar
          </h3>
          
          <p className="text-gray-500 dark:text-gray-400 mb-4">
            Suporte para múltiplos arquivos PDF
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            multiple
            onChange={(e) => handleFileSelect(e.target.files)}
            className="hidden"
          />

          <Button
            onClick={() => fileInputRef.current?.click()}
            icon={<Upload size={16} />}
          >
            Selecionar Arquivos
          </Button>
        </div>
      </Card>

      {/* Upload Progress */}
      <AnimatePresence>
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Arquivos Carregados ({uploadedFiles.length})
                </h3>
              </div>

              <div className="space-y-3">
                {uploadedFiles.map((uploadFile, index) => (
                  <motion.div
                    key={`${uploadFile.file.name}-${index}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg"
                  >
                    <div className="flex items-center space-x-3 flex-1 min-w-0">
                      {getStatusIcon(uploadFile.status)}
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                            {uploadFile.file.name}
                          </p>
                          
                          <Badge variant={getStatusColor(uploadFile.status)} size="sm">
                            {uploadFile.status === 'uploading' && 'Enviando'}
                            {uploadFile.status === 'processing' && 'Processando'}
                            {uploadFile.status === 'success' && 'Concluído'}
                            {uploadFile.status === 'error' && 'Erro'}
                          </Badge>
                        </div>

                        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                          <span>{formatFileSize(uploadFile.file.size)}</span>
                          
                          {uploadFile.response?.chunks_created && (
                            <span>{uploadFile.response.chunks_created} chunks criados</span>
                          )}
                        </div>

                        {/* Progress Bar */}
                        {(uploadFile.status === 'uploading' || uploadFile.status === 'processing') && (
                          <div className="mt-2">
                            <div className="flex items-center justify-between text-xs mb-1">
                              <span>Progresso</span>
                              <span>{uploadFile.progress}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5">
                              <div
                                className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                                style={{ width: `${uploadFile.progress}%` }}
                              />
                            </div>
                          </div>
                        )}

                        {/* Error Message */}
                        {uploadFile.error && (
                          <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                            {uploadFile.error}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center space-x-2 ml-3">
                      {uploadFile.status === 'error' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => retryUpload(uploadFile)}
                          icon={<RefreshCw size={12} />}
                        >
                          Tentar Novamente
                        </Button>
                      )}

                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(uploadFile.file)}
                        icon={<X size={12} />}
                      />
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Reindex Progress */}
      <AnimatePresence>
        {(isReindexing || reindexProgress.length > 0) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                  <RefreshCw size={18} className={`mr-2 ${isReindexing ? 'animate-spin' : ''}`} />
                  Reindexação de Documentos
                </h3>
                
                {isReindexing && (
                  <Badge variant="warning" size="sm">
                    Em Progresso
                  </Badge>
                )}
              </div>

              <div className="space-y-2 max-h-60 overflow-y-auto">
                {reindexProgress.map((update, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-2 text-sm bg-gray-50 dark:bg-gray-900 rounded"
                  >
                    <span className="text-gray-700 dark:text-gray-300">
                      {update.message}
                    </span>
                    
                    <div className="flex items-center space-x-2">
                      {update.total_files && (
                        <span className="text-xs text-gray-500">
                          {update.total_files} arquivos
                        </span>
                      )}
                      
                      <Badge
                        variant={
                          update.type === 'complete' ? 'success' :
                          update.type === 'error' ? 'error' : 'info'
                        }
                        size="sm"
                      >
                        {update.type}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>

              {!isReindexing && reindexProgress.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setReindexProgress([])}
                  >
                    Limpar Log
                  </Button>
                </div>
              )}
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Help Text */}
      <Card className="bg-blue-50 dark:bg-blue-900/20">
        <div className="flex items-start space-x-3">
          <FileText size={20} className="text-blue-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-1">
              Dicas para Upload
            </h4>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
              <li>• Carregue documentos técnicos em português ou inglês</li>
              <li>• PDFs com definições e siglas têm prioridade maior na busca</li>
              <li>• Arquivos são processados automaticamente e indexados</li>
              <li>• Use a reindexação após carregar vários documentos</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PDFUpload;