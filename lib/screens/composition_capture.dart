import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import '../services/analysis_service.dart';

class CompositionCapture extends StatefulWidget {
  const CompositionCapture({super.key});

  @override
  State<CompositionCapture> createState() => _CompositionCaptureState();
}

class _CompositionCaptureState extends State<CompositionCapture> {
  CameraController? _controller;
  bool _isCameraInitialized = false;
  bool _isCapturing = false;
  bool _isAnalyzing = false;
  String? _capturedImagePath;
  final AnalysisService _analysisService = AnalysisService();
  Map<String, dynamic>? _analysisResults;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      return;
    }

    _controller = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _captureImage() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      final XFile image = await _controller!.takePicture();
      final Directory appDir = await getApplicationDocumentsDirectory();
      final String fileName =
          'composition_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final String filePath = path.join(appDir.path, fileName);

      // Copy the image to app directory
      await File(image.path).copy(filePath);

      setState(() {
        _capturedImagePath = filePath;
        _isCapturing = false;
        _isAnalyzing = true;
      });

      // Analyze the image
      try {
        final results =
            await _analysisService.analyzeComposition(File(filePath));

        setState(() {
          _analysisResults = results;
          _isAnalyzing = false;
        });

        _showAnalysisResults();
      } catch (e) {
        setState(() {
          _isAnalyzing = false;
        });
        _showError('Failed to analyze composition: $e');
      }
    } catch (e) {
      debugPrint('Error capturing image: $e');
      setState(() {
        _isCapturing = false;
        _isAnalyzing = false;
      });
      _showError('Failed to capture image: $e');
    }
  }

  void _showAnalysisResults() {
    if (_analysisResults == null) return;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Composition Analysis Results'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Material Type: ${_analysisResults!['material_type']}'),
            Text(
                'Transparency: ${_analysisResults!['transparency'].toStringAsFixed(2)}'),
            Text(
                'Homogeneity: ${_analysisResults!['homogeneity'].toStringAsFixed(2)}'),
            Text(
                'Material Quality: ${_analysisResults!['material_quality'].toStringAsFixed(2)}'),
            Text(
                'Spectral Confidence: ${_analysisResults!['spectral_confidence'].toStringAsFixed(2)}'),
            Text(
                'Overall Confidence: ${_analysisResults!['confidence'].toStringAsFixed(2)}'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showError(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Composition Analysis'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                CameraPreview(_controller!),
                Positioned.fill(
                  child: CustomPaint(
                    painter: CompositionGuidePainter(),
                  ),
                ),
                if (_isAnalyzing)
                  Container(
                    color: Colors.black54,
                    child: const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(height: 16),
                          Text(
                            'Analyzing Composition...',
                            style: TextStyle(color: Colors.white),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.fromLTRB(16.0, 16.0, 16.0, 116.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Position the lens against a white background for best results',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed:
                      (_isCapturing || _isAnalyzing) ? null : _captureImage,
                  child: _isCapturing
                      ? const CircularProgressIndicator()
                      : const Text('Capture Image'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class CompositionGuidePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Draw center rectangle for lens placement
    final center = Offset(size.width / 2, size.height / 2);
    final width = size.width * 0.6;
    final height = size.height * 0.4;

    final rect = Rect.fromCenter(
      center: center,
      width: width,
      height: height,
    );

    canvas.drawRect(rect, paint);

    // Draw diagonal lines for alignment
    canvas.drawLine(
      Offset(rect.left, rect.top),
      Offset(rect.right, rect.bottom),
      paint,
    );
    canvas.drawLine(
      Offset(rect.right, rect.top),
      Offset(rect.left, rect.bottom),
      paint,
    );
  }

  @override
  bool shouldRepaint(CompositionGuidePainter oldDelegate) => false;
}
