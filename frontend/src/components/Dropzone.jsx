import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function Dropzone({ selectedFile, previewUrl, onFileSelected, onClear }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelected(acceptedFiles[0]);
    }
  }, [onFileSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpeg', '.jpg'],
      'image/png': ['.png'],
      'image/webp': ['.webp']
    },
    maxSize: 10 * 1024 * 1024,
    multiple: false
  });

  return (
    <div 
      {...getRootProps()}
      className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all relative overflow-hidden group min-h-[220px]
        ${isDragActive ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-300 hover:border-indigo-500/70 hover:bg-indigo-50/20 dark:border-slate-800 dark:hover:bg-slate-950/20'}
      `}
    >
      <input {...getInputProps()} />
      
      {!selectedFile ? (
        <div className="text-center flex flex-col items-center space-y-4">
          <div className="h-12 w-12 rounded-xl bg-slate-100 flex items-center justify-center text-slate-400 group-hover:text-indigo-400 group-hover:scale-110 transition-all dark:bg-slate-950/80">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
            </svg>
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              {isDragActive ? "Drop the file here..." : "Drag & drop your image here"}
            </p>
            <p className="text-xs text-slate-400 mt-1 dark:text-slate-500">or click to browse local files</p>
          </div>
          <span className="inline-block px-2.5 py-1 text-[9px] font-bold text-slate-400 bg-slate-100 rounded dark:bg-slate-950 dark:text-slate-500">
            JPG, PNG, WEBP — Max 10MB
          </span>
        </div>
      ) : (
        <div className="w-full h-full flex flex-col items-center relative" onClick={(e) => e.stopPropagation()}>
          <img 
            className="max-h-72 w-full object-contain rounded-lg shadow-md" 
            src={previewUrl} 
            alt="Preview" 
          />
          <button 
            onClick={onClear}
            className="absolute top-2 right-2 p-1.5 rounded-lg bg-slate-950/80 text-slate-400 hover:text-white transition-all shadow-md"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>
      )}
    </div>
  )
}

export default Dropzone
