import { z } from 'zod';

// File validation schema
export const fileUploadSchema = z.object({
  name: z.string().min(1).max(255),
  size: z.number().positive(),
  type: z.string().refine(
    (type) => type === 'application/zip' || type === 'application/x-zip-compressed',
    'Apenas arquivos ZIP são permitidos'
  ),
});

// URL upload schema
export const urlUploadSchema = z.object({
  url: z.string().url('URL inválida').max(2048),
});

// Document tag schema
export const documentTagSchema = z.object({
  name: z.string().trim().min(1).max(50),
  color: z.string().regex(/^#[0-9A-F]{6}$/i, 'Cor inválida'),
});

// Document note schema
export const documentNoteSchema = z.object({
  content: z.string().trim().min(1).max(1000),
  documentId: z.string().uuid(),
});

// XML content validation
export const validateXMLStructure = (content: string): boolean => {
  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(content, 'text/xml');
    const parseError = doc.querySelector('parsererror');
    return !parseError;
  } catch {
    return false;
  }
};

// Check for duplicate files
export const isDuplicateFile = (
  newFile: { name: string; size: number },
  existingFiles: { name: string; size: number }[]
): boolean => {
  return existingFiles.some(
    (file) => file.name === newFile.name && file.size === newFile.size
  );
};

// Sanitize filename
export const sanitizeFilename = (filename: string): string => {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .substring(0, 255);
};
