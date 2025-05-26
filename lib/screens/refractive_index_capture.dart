import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import '../services/analysis_service.dart';

class RefractiveIndexCapture extends StatefulWidget {
  const RefractiveIndexCapture({super.key});

  @override
  State<RefractiveIndexCapture> createState() => _RefractiveIndexCaptureState();
}

class _RefractiveIndexCaptureState extends State<RefractiveIndexCapture> {
  CameraController? _controller;
  bool _isCameraInitialized = false;
  bool _isCapturing = false;
  bool _isAnalyzing = false;
  String? _capturedImagePath;
  double _currentAngle = 45.0;
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
          'refractive_index_${DateTime.now().millisecondsSinceEpoch}.jpg';
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
        final results = await _analysisService.analyzeRefractiveIndex(
          File(filePath),
          _currentAngle,
        );

        setState(() {
          _analysisResults = results;
          _isAnalyzing = false;
        });

        _showAnalysisResults();
      } catch (e) {
        setState(() {
          _isAnalyzing = false;
        });
        _showError('Failed to analyze image: $e');
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
        title: const Text('Analysis Results'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
                'Reflectance Score: ${_analysisResults!['reflectance_score'].toStringAsFixed(2)}'),
            Text(
                'Confidence: ${_analysisResults!['confidence'].toStringAsFixed(2)}'),
            Text(
                'Estimated Refractive Index: ${_analysisResults!['estimated_refractive_index'].toStringAsFixed(2)}'),
            Text(
                'Quality Score: ${_analysisResults!['quality_score'].toStringAsFixed(2)}'),
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
        title: const Text('Refractive Index Analysis'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                CameraPreview(_controller!),
                Positioned.fill(
                  child: CustomPaint(
                    painter: CaptureGuidePainter(angle: _currentAngle),
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
                            'Analyzing Image...',
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
                Text(
                  'Angle: ${_currentAngle.toStringAsFixed(1)}°',
                  style: Theme.of(context).textTheme.titleLarge,
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

class CaptureGuidePainter extends CustomPainter {
  final double angle;

  CaptureGuidePainter({required this.angle});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Draw angle guide
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width * 0.4;

    // Draw angle line
    final angleRadians = angle * (pi / 180);
    final endPoint = Offset(
      center.dx + radius * cos(angleRadians),
      center.dy + radius * sin(angleRadians),
    );

    canvas.drawLine(center, endPoint, paint);

    // Draw angle arc
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius * 0.8),
      0,
      angleRadians,
      false,
      paint,
    );
  }

  @override
  bool shouldRepaint(CaptureGuidePainter oldDelegate) {
    return oldDelegate.angle != angle;
  }
}
