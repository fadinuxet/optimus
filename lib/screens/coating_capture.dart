import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import '../services/analysis_service.dart';
import 'dart:convert';

class CoatingCapture extends StatefulWidget {
  const CoatingCapture({super.key});

  @override
  State<CoatingCapture> createState() => _CoatingCaptureState();
}

class _CoatingCaptureState extends State<CoatingCapture> {
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
          'coating_${DateTime.now().millisecondsSinceEpoch}.jpg';
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
        final results = await _analysisService.analyzeCoating(File(filePath));
        setState(() {
          _analysisResults = results;
          _isAnalyzing = false;
        });
        print('Analysis Results: $_analysisResults');
        _showAnalysisResults();
      } catch (e) {
        setState(() {
          _isAnalyzing = false;
        });
        _showError('Failed to analyze coating: $e');
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
        title: const Text('Coating Analysis Results'),
        content: _analysisResults == null
            ? const Text('No analysis explanations available.')
            : Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (var item in _analysisResults!.entries)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(item.key,
                              style:
                                  const TextStyle(fontWeight: FontWeight.bold)),
                          Text('Score: ${item.value}'),
                          if (item.value is Map<String, dynamic> &&
                              item.value['label'] != null)
                            Text('Label: ${item.value['label']}'),
                          Text(item.value is Map<String, dynamic>
                              ? (item.value['description'] ?? '')
                              : ''),
                        ],
                      ),
                    ),
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
        title: const Text('Coating Analysis'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                CameraPreview(_controller!),
                Positioned.fill(
                  child: CustomPaint(
                    painter: CoatingGuidePainter(),
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
                            'Analyzing Coating...',
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
                  'Position the lens in direct light to analyze coating',
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

class CoatingGuidePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Draw center circle for lens placement
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width * 0.3;

    canvas.drawCircle(center, radius, paint);

    // Draw crosshair
    canvas.drawLine(
      Offset(center.dx - radius, center.dy),
      Offset(center.dx + radius, center.dy),
      paint,
    );
    canvas.drawLine(
      Offset(center.dx, center.dy - radius),
      Offset(center.dx, center.dy + radius),
      paint,
    );
  }

  @override
  bool shouldRepaint(CoatingGuidePainter oldDelegate) => false;
}
