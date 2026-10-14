"""Small explicit entry point; configure before importing any workflow modules."""

import argparse
import json
import logging
import os


def main():
    parser = argparse.ArgumentParser(prog="lavira_m3")
    parser.add_argument("--config", help="fleet JSON configuration")
    parser.add_argument("--ros", choices=("ros1", "ros2"), default="ros2")
    parser.add_argument("--agent", help="configured agent identifier")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="validate and print non-secret fleet configuration")
    hub = sub.add_parser("hub", help="shared maps, target board and web monitor")
    hub.add_argument("--host", default="127.0.0.1")
    hub.add_argument("--port", type=int, default=8770)
    sub.add_parser("map-bridge", help="registered cloud upload and shared planning maps")
    sub.add_parser("simulate", help="synthetic connectivity fixture; no hardware")
    for name in ("agent", "navigate", "gate", "navigation-server"):
        command = sub.add_parser(name)
        command.add_argument("--control", action="store_true")
        if name == "agent":
            command.add_argument("--goals")
            command.add_argument("--replay")
        if name == "navigate":
            command.add_argument("xy", type=float, nargs=2)
    args = parser.parse_args()
    if args.config:
        os.environ["NAV_CONFIG"] = args.config
    if args.agent:
        os.environ["NAV_AGENT"] = args.agent
    os.environ["NAV_ROS"] = args.ros
    from .profiles import AGENTS, backend
    from .settings import AGENT, SERVICES

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.command == "config":
        print(
            json.dumps(
                dict(
                    agents={
                        name: dict(profile, navigation_backend=backend(name))
                        for name, profile in AGENTS.items()
                    },
                    ros=args.ros,
                ),
                indent=2,
            )
        )
        return
    if args.command == "hub":
        from .hub import serve

        serve(args.host, args.port)
        return
    if args.command == "simulate":
        from .simulation import run

        run(args.ros)
        return
    if args.command == "map-bridge":
        from .map_bridge import MapBridge

        MapBridge(args.ros).run()
        return
    if args.command in ("gate", "navigation-server"):
        if args.ros != "ros2":
            parser.error("ROS1 gate is integrated in its navigation client")
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.signals import SignalHandlerOptions

        from .adapters.ros2.gate import Gate
        from .adapters.ros2.navigation_server import Navigation

        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = Gate(args.control) if args.command == "gate" else Navigation(args.control)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            if hasattr(node, "close"):
                node.close()
            node.destroy_node()
            rclpy.try_shutdown()
        return
    from .runtime import Runtime

    runtime = Runtime(args.ros, args.control)
    try:
        if args.command == "navigate":
            if not args.control:
                parser.error("navigate requires explicit --control")
            result = runtime.navigate_to(*args.xy)
            print(json.dumps(result.as_dict()))
            if not result.ok:
                raise SystemExit(1)
        else:
            from .decision.agent_loop import LaviraM3Agent
            from .decision.config import RobotConfig, load_goals
            from .decision.replay import ReplayRouter

            router = ReplayRouter(args.replay) if args.replay else None
            robot = RobotConfig(name=AGENT, hub=SERVICES["board"], nav_url=SERVICES["map"])
            agent = LaviraM3Agent(
                robot,
                runtime,
                load_goals(args.goals) if args.goals else [],
                llm=router,
                from_board=not args.goals,
            )
            agent.run()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
